import os
import gc
import time
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from step3_dataset_det import (
    VOCDetectionDataset, detection_collate_fn, build_index_dataframe,
    build_train_transform, build_val_transform, IMG_W, IMG_H,
)
from step3_models_det import build_faster_rcnn, build_ssd
from step3_utils_det import (
    plot_confusion_matrix, plot_training_curves,
    compute_detection_metrics, compute_f1_from_confusion,
    count_params, measure_fps, measure_flops,
    wilson_ci, bootstrap_f1_ci, subgroup_metrics,
)
from step3_yolov7_runner import (
    prepare_yolo_dataset, write_data_yaml, run_yolov7x,
    parse_yolov7_results, run_yolov7_test_fps,
)

# ---------------- 全局配置 ----------------
SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 150
BATCH_SIZE = 16
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EARLY_STOP_PATIENCE = 15
NUM_CLASSES = 3

VOC_ROOT = r'/root/workspace/tooth/data/VOCdevkit/tooth'
YOLOV7_REPO = r'/root/workspace/tooth/external/yolov7'
YOLOV7_WEIGHTS = os.path.join(YOLOV7_REPO, 'yolov7x.pt')   # 官方预训练权重
YOLO_IMG_SIZE = 640                                        # 官方默认


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def create_output_dir(base_path):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out = os.path.join(base_path, f'output/results_{ts}')
    os.makedirs(out, exist_ok=True)
    return out


# ---------------- torchvision 训练/推理 ----------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss = sum(model(images, targets).values())
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate_loss(model, loader, device):
    """检测模型在 eval 下不返回 loss，用 train 模式前向仅取 loss。"""
    model.train()
    total = 0.0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        total += sum(model(images, targets).values()).item()
    model.eval()
    return total / max(len(loader), 1)


@torch.no_grad()
def predict(model, loader, device, score_thresh=0.05):
    model.eval()
    pb, pl, ps, gb, gl = [], [], [], [], []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        for out, tgt in zip(model(images), targets):
            keep = out['scores'] >= score_thresh
            pb.append(out['boxes'][keep].cpu().numpy())
            pl.append(out['labels'][keep].cpu().numpy())
            ps.append(out['scores'][keep].cpu().numpy())
            gb.append(tgt['boxes'].numpy())
            gl.append(tgt['labels'].numpy())
    return pb, pl, ps, gb, gl


def top1_predictions(pl, ps, gl):
    """每图取 top-1 预测标签 / GT 标签（单 ROI 任务）。"""
    y_true, y_pred = [], []
    for tgt, labs, scs in zip(gl, pl, ps):
        y_true.append(int(tgt[0]) if len(tgt) else 0)
        y_pred.append(int(labs[np.argmax(scs)]) if len(labs) else 0)
    return y_true, y_pred


# ---------------- 单个 torchvision 模型全流程 ----------------
def run_torchvision_model(model_name, builder, loaders, test_df,
                          output_dir, device):
    train_loader, val_loader, test_loader = loaders
    print(f'\n===== {model_name} =====')

    model = builder(num_classes=NUM_CLASSES, pretrained=True).to(device)
    params_m = count_params(model) / 1e6
    flops = measure_flops(model, input_size=(1, 3, IMG_H, IMG_W), device=device)

    optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    tr_losses, va_losses = [], []
    best_val, patience = float('inf'), 0

    for epoch in range(NUM_EPOCHS):
        tr = train_one_epoch(model, train_loader, optimizer, device)
        va = validate_loss(model, val_loader, device)
        scheduler.step()
        tr_losses.append(tr); va_losses.append(va)
        print(f'Epoch {epoch+1:3d} | train {tr:.4f} | val {va:.4f}')

        if va < best_val:
            best_val = va; patience = 0
            torch.save(model.state_dict(),
                       f'{output_dir}/best_{model_name}.pth')
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f'Early stop @ epoch {epoch+1}')
                break

    plot_training_curves(tr_losses, va_losses,
                         f'{output_dir}/{model_name}_loss.png')

    # 测试
    model.load_state_dict(torch.load(f'{output_dir}/best_{model_name}.pth'))
    pb, pl, ps, gb, gl = predict(model, test_loader, device)
    det = compute_detection_metrics(pb, pl, ps, gb, gl, iou_threshold=0.5)

    fps, n_img, sec = measure_fps(model, test_loader, device,
                                  warmup=3, max_batches=None)

    y_true, y_pred = top1_predictions(pl, ps, gl)
    f1m = compute_f1_from_confusion(y_true, y_pred)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == p and t != 0)
    acc_ci = wilson_ci(tp, len(y_true))
    f1_ci = bootstrap_f1_ci(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred,
                          f'{output_dir}/{model_name}_cm.png')

    # 亚组
    tdf = test_df.reset_index(drop=True).copy()
    tdf['y_true'] = y_true; tdf['y_pred'] = y_pred
    subgroup_metrics(tdf).to_csv(
        f'{output_dir}/{model_name}_subgroup.csv', index=False)

    # 保存预测明细
    pd.DataFrame({
        'image_path': tdf['image_path'],
        'subgroup': tdf['subgroup'],
        'y_true': y_true, 'y_pred': y_pred,
    }).to_csv(f'{output_dir}/{model_name}_predictions.csv', index=False)

    del model; gc.collect(); torch.cuda.empty_cache()

    return {
        'model': model_name,
        'params(M)': params_m,
        'FLOPs(G)': flops / 1e9 if flops > 0 else -1,
        'FPS': fps,
        'AP_fill': det['AP_fill'],
        'AP_rct': det['AP_rct'],
        'mAP@0.5': det['mAP'],
        'Precision': f1m['Precision'],
        'Recall': f1m['Recall'],
        'F1': f1m['F1'],
        'F1_95CI_low': f1_ci[0], 'F1_95CI_high': f1_ci[1],
        'Acc_95CI_low': acc_ci[0], 'Acc_95CI_high': acc_ci[1],
    }


# ---------------- YOLOv7-X 全流程 ----------------
def run_yolov7x_pipeline(df_with_split, output_dir):
    print('\n===== YOLOv7-X =====')

    yolo_root = os.path.join(output_dir, 'yolo_dataset')
    prepare_yolo_dataset(df_with_split, yolo_root, W=IMG_W, H=IMG_H)
    data_yaml = write_data_yaml(yolo_root, os.path.join(output_dir, 'data.yaml'))

    run_dir = run_yolov7x(
        repo_root=YOLOV7_REPO,
        data_yaml=data_yaml,
        output_dir=output_dir,
        weights=YOLOV7_WEIGHTS,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        img_size=YOLO_IMG_SIZE,
        device='0',
    )

    metrics = parse_yolov7_results(run_dir)
    fps = run_yolov7_test_fps(
        YOLOV7_REPO,
        weights=os.path.join(run_dir, 'weights', 'best.pt'),
        data_yaml=data_yaml,
        img_size=YOLO_IMG_SIZE,
        batch_size=1,
        device='0',
    )

    return {
        'model': 'YOLOv7-X',
        'params(M)': 71.3,          # 官方 yolov7x 约 71.3M
        'FLOPs(G)': 189.9,          # 官方 640 输入约 189.9 GFLOPs
        'FPS': fps if fps else -1,
        'AP_fill': np.nan,
        'AP_rct': np.nan,
        'mAP@0.5': metrics.get('mAP@0.5', np.nan),
        'Precision': metrics.get('precision', np.nan),
        'Recall': metrics.get('recall', np.nan),
        'F1': np.nan,
        'F1_95CI_low': np.nan, 'F1_95CI_high': np.nan,
        'Acc_95CI_low': np.nan, 'Acc_95CI_high': np.nan,
    }


# ---------------- 主流程 ----------------
def main():
    set_seed()
    print('Device:', DEVICE)

    # 1. 生成/读取索引
    index_csv = os.path.join(VOC_ROOT, 'detection_index.csv')
    if os.path.exists(index_csv):
        df = pd.read_csv(index_csv)
    else:
        df = build_index_dataframe(VOC_ROOT)
        df.to_csv(index_csv, index=False)
    print(f'Total images: {len(df)}')
    print(df['label'].value_counts().to_dict())

    base_path = os.path.dirname(VOC_ROOT)
    output_dir = create_output_dir(base_path)

    # 2. 患者级 8:1:1 分层划分（每患者 1 张片 -> 分层即患者级）
    train_val, test = train_test_split(
        df, test_size=0.1, random_state=SEED, stratify=df['label'])
    train, val = train_test_split(
        train_val, test_size=0.1 / 0.9, random_state=SEED,
        stratify=train_val['label'])
    print(f'train={len(train)} val={len(val)} test={len(test)}')

    # 保存划分，便于复现
    for name, sub in [('train', train), ('val', val), ('test', test)]:
        sub[['image_path', 'xml_path', 'label', 'subgroup']] \
            .to_csv(os.path.join(output_dir, f'split_{name}.csv'), index=False)

    # 3. torchvision DataLoader
    train_loader = DataLoader(
        VOCDetectionDataset(train, transform=build_train_transform()),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
        collate_fn=detection_collate_fn, pin_memory=True)
    val_loader = DataLoader(
        VOCDetectionDataset(val, transform=build_val_transform()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        collate_fn=detection_collate_fn, pin_memory=True)
    test_loader = DataLoader(
        VOCDetectionDataset(test, transform=build_val_transform()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
        collate_fn=detection_collate_fn, pin_memory=True)
    loaders = (train_loader, val_loader, test_loader)

    all_results = []

    # 4. Faster R-CNN + SSD
    for model_name, builder in [('FasterRCNN', build_faster_rcnn),
                                ('SSD', build_ssd)]:
        res = run_torchvision_model(model_name, builder, loaders,
                                    test, output_dir, DEVICE)
        all_results.append(res)

    # 5. YOLOv7-X
    df_with_split = df.copy()
    df_with_split.loc[train.index, 'split'] = 'train'
    df_with_split.loc[val.index,   'split'] = 'val'
    df_with_split.loc[test.index,  'split'] = 'test'
    try:
        res = run_yolov7x_pipeline(df_with_split, output_dir)
        all_results.append(res)
    except Exception as e:
        print(f'[YOLOv7X] failed: {e}')

    # 6. 汇总
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    print('\n===== 汇总 =====')
    print(res_df.to_string(index=False))


if __name__ == '__main__':
    main()
