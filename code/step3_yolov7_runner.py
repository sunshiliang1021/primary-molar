import os
import subprocess
import shutil
import yaml
import pandas as pd
import numpy as np


def _yolo_label_line(cls_id, xmin, ymin, xmax, ymax, W, H):
    """YOLO 格式: cls cx cy w h（归一化）。cls 0/1 对应 fill/rct。"""
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w = (xmax - xmin) / W
    h = (ymax - ymin) / H
    return f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def prepare_yolo_dataset(df, dst_root, W=620, H=480):
    """
    将 VOC (image + xml) 转成 YOLOv7 需要的目录结构：
        dst_root/images/{train,val,test}/*.jpg
        dst_root/labels/{train,val,test}/*.txt
    df 需含列: image_path, xml_path, split ('train'/'val'/'test')
    同时读取 XML 里的 bbox。
    """
    import xml.etree.ElementTree as ET
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dst_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dst_root, 'labels', split), exist_ok=True)

    name_map = {'fill': 0, 'rct': 1}
    for _, row in df.iterrows():
        split = row['split']
        img_src = row['image_path']
        xml_src = row['xml_path']
        stem = os.path.splitext(os.path.basename(img_src))[0]
        img_dst = os.path.join(dst_root, 'images', split, stem + '.jpg')
        lab_dst = os.path.join(dst_root, 'labels', split, stem + '.txt')

        shutil.copyfile(img_src, img_dst)

        root = ET.parse(xml_src).getroot()
        lines = []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name not in name_map:
                continue
            b = obj.find('bndbox')
            lines.append(_yolo_label_line(
                name_map[name],
                float(b.find('xmin').text), float(b.find('ymin').text),
                float(b.find('xmax').text), float(b.find('ymax').text),
                W, H,
            ))
        with open(lab_dst, 'w') as f:
            f.write('\n'.join(lines))


def write_data_yaml(dst_root, save_path):
    data = {
        'train': os.path.join(dst_root, 'images', 'train'),
        'val':   os.path.join(dst_root, 'images', 'val'),
        'test':  os.path.join(dst_root, 'images', 'test'),
        'nc': 2,
        'names': ['fill', 'rct'],
    }
    with open(save_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return save_path


def run_yolov7x(repo_root, data_yaml, output_dir,
                weights=None, batch_size=16, epochs=150,
                img_size=640, device='0'):
    """
    调用官方 yolov7/train.py 训练 YOLOv7-X。
    - weights: 预训练权重路径（推荐 yolov7x.pt，ImageNet 预训练迁移）
    - 官方脚本要求的 cwd 是 repo_root
    """
    train_py = os.path.join(repo_root, 'train.py')
    cfg = os.path.join(repo_root, 'cfg', 'training', 'yolov7x.yaml')
    hyp = os.path.join(repo_root, 'data', 'hyp.scratch.p5.yaml')
    if not os.path.exists(hyp):
        hyp = os.path.join(repo_root, 'data', 'hyp.scratch.yaml')

    cmd = [
        'python', train_py,
        '--workers', '4',
        '--device', device,
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--img-size', str(img_size), str(img_size),
        '--data', data_yaml,
        '--hyp', hyp,
        '--cfg', cfg,
        '--name', 'yolov7x_tooth',
        '--project', output_dir,
        '--exist-ok',
    ]
    if weights and os.path.exists(weights):
        cmd += ['--weights', weights]

    print('[YOLOv7X] cmd:', ' '.join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)

    run_dir = os.path.join(output_dir, 'yolov7x_tooth')
    return run_dir


def parse_yolov7_results(run_dir):
    """读取官方训练输出的 results.csv，返回最后一行/最优 epoch 指标。"""
    csv_path = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # 官方列名示例: epoch, train/box_loss, ..., metrics/mAP_0.5, metrics/mAP_0.5:0.95
    best = df.loc[df['metrics/mAP_0.5'].idxmax()] \
        if 'metrics/mAP_0.5' in df.columns else df.iloc[-1]
    return {
        'mAP@0.5': float(best.get('metrics/mAP_0.5', np.nan)),
        'mAP@0.5:0.95': float(best.get('metrics/mAP_0.5:0.95', np.nan)),
        'precision': float(best.get('metrics/precision', np.nan)),
        'recall': float(best.get('metrics/recall', np.nan)),
    }


def run_yolov7_test_fps(repo_root, weights, data_yaml,
                        img_size=640, batch_size=1, device='0'):
    """
    用官方 test.py 推理一次，读取其 FPS 输出（若解析失败则返回 None）。
    官方 test.py 会打印 'Speed: x.x/xx.x ms inference, ...'。
    """
    test_py = os.path.join(repo_root, 'test.py')
    cmd = [
        'python', test_py,
        '--data', data_yaml,
        '--img-size', str(img_size),
        '--batch-size', str(batch_size),
        '--device', device,
        '--weights', weights,
        '--task', 'val',
        '--name', 'yolov7x_fps',
    ]
    print('[YOLOv7X-test] cmd:', ' '.join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True,
                          text=True, check=False)
    out = proc.stdout + proc.stderr
    # 解析 "Speed: 1.2/3.4 ms inference"
    import re
    m = re.search(r'Speed:\s*([\d.]+)/([\d.]+)\s*ms', out)
    if m:
        infer_ms = float(m.group(1))
        return 1000.0 / infer_ms
    return None
