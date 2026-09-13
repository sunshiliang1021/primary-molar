import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# ---------- 可视化 ----------
def plot_confusion_matrix(y_true, y_pred, save_path, class_names=('fill', 'rct')):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


def plot_training_curves(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training / Validation Loss')
    plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path, dpi=300); plt.close()


# ---------- IoU / mAP ----------
def iou_xyxy(a, b):
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def compute_detection_metrics(pred_boxes_list, pred_labels_list, pred_scores_list,
                              gt_boxes_list, gt_labels_list,
                              iou_threshold=0.5, num_classes=2):
    """
    单框检测 mAP（VOC 2010+ 11 点插值）。
    每图取 top-1 预测框，与 GT 单框做 IoU 匹配。
    """
    aps = {}
    for cls in range(1, num_classes + 1):
        records, n_gt = [], 0
        for pb, pl, ps, gb, gl in zip(pred_boxes_list, pred_labels_list,
                                      pred_scores_list,
                                      gt_boxes_list, gt_labels_list):
            gt_cls = gb[gl == cls]
            n_gt += len(gt_cls)
            m = pl == cls
            pb_c, ps_c = pb[m], ps[m]
            if len(pb_c) == 0:
                continue
            k = int(np.argmax(ps_c))
            box, score = pb_c[k], ps_c[k]
            if len(gt_cls) == 0:
                records.append((score, 0))
            else:
                best = max(iou_xyxy(box, g) for g in gt_cls)
                records.append((score, int(best >= iou_threshold)))
        if n_gt == 0 or len(records) == 0:
            aps[cls] = 0.0
            continue
        records.sort(key=lambda x: -x[0])
        tp = np.array([r[1] for r in records])
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(1 - tp)
        recall = cum_tp / (n_gt + 1e-9)
        precision = cum_tp / (cum_tp + cum_fp + 1e-9)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p = precision[recall >= t].max() if np.any(recall >= t) else 0.0
            ap += p / 11
        aps[cls] = ap
    return {'AP_fill': aps.get(1, 0.0),
            'AP_rct': aps.get(2, 0.0),
            'mAP': float(np.mean(list(aps.values())))}


def compute_f1_from_confusion(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1, 2], average='macro', zero_division=0)
    return {'Precision': float(p), 'Recall': float(r), 'F1': float(f1)}


# ---------- 效率 ----------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_fps(model, loader, device, warmup=3, max_batches=None):
    """
    推理时间统计 FPS：warmup 后统计若干 batch 的平均耗时。
    返回 (fps, n_images, elapsed_sec)
    """
    model.eval()
    n_img, t0 = 0, None
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            images = [img.to(device) for img in images]
            if i < warmup:
                _ = model(images)
                continue
            if t0 is None:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
            _ = model(images)
            n_img += len(images)
            if max_batches and (i - warmup + 1) >= max_batches:
                break
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    return n_img / (elapsed + 1e-9), n_img, elapsed


def measure_flops(model, input_size=(1, 3, 480, 620), device='cuda'):
    """用 thop 统计 FLOPs；失败返回 -1。"""
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except Exception as e:
        print(f'[FLOPs] failed: {e}')
        return -1


# ---------- 统计 ----------
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_f1_ci(y_true, y_pred, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true[idx], y_pred[idx], labels=[1, 2],
            average='macro', zero_division=0)
        vals.append(f1)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def subgroup_metrics(df, y_true_col='y_true', y_pred_col='y_pred',
                     group_col='subgroup'):
    rows = []
    for g in df[group_col].unique():
        sub = df[df[group_col] == g]
        m = compute_f1_from_confusion(sub[y_true_col].values,
                                      sub[y_pred_col].values)
        rows.append({'subgroup': g, 'n': len(sub), **m})
    return pd.DataFrame(rows)
