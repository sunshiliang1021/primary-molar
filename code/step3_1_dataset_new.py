import os
import numpy as np
import pandas as pd
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_MAP = {'fill': 1, 'rct': 2}          # 0 = background
CLASS_NAMES = ['__background__', 'fill', 'rct']
IMG_W, IMG_H = 620, 480


# ---------------- albumentations 变换 ----------------
def build_train_transform():
    """训练集：随机水平翻转(50%) + 随机旋转(±15°) + 亮度/对比度调整。"""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=0, p=0.5),   # ±15°，黑色填充
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels'],
            min_visibility=0.3),
    )


def build_val_transform():
    """验证/测试集：仅 resize + ToTensor，不做增强。"""
    return A.Compose(
        [],
        bbox_params=A.BboxParams(
            format='pascal_voc', label_fields=['labels']),
    )


# ---------------- Dataset ----------------
class VOCDetectionDataset(Dataset):
    """
    乳牙根尖片检测数据集（PASCAL VOC XML，每图 1 个 ROI）。
    CSV 需含: image_path, xml_path, label, subgroup
    """

    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _parse_xml(xml_path):
        root = ET.parse(xml_path).getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            name = obj.find('name').text.strip().lower()
            if name not in CLASS_MAP:
                continue
            b = obj.find('bndbox')
            xmin = float(b.find('xmin').text)
            ymin = float(b.find('ymin').text)
            xmax = float(b.find('xmax').text)
            ymax = float(b.find('ymax').text)
            # 边界保护
            xmin = max(0.0, min(xmin, IMG_W - 1))
            ymin = max(0.0, min(ymin, IMG_H - 1))
            xmax = max(xmin + 1.0, min(xmax, IMG_W))
            ymax = max(ymin + 1.0, min(ymax, IMG_H))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CLASS_MAP[name])
        if not boxes:
            return (np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        return (np.asarray(boxes, dtype=np.float32),
                np.asarray(labels, dtype=np.int64))

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img = Image.open(row['image_path']).convert('RGB')
        img = img.resize((IMG_W, IMG_H))
        img_np = np.array(img)                          # HWC, uint8

        boxes, labels = self._parse_xml(row['xml_path'])

        if self.transform is not None:
            out = self.transform(image=img_np, bboxes=boxes,
                                 labels=labels.tolist())
            img_np = out['image']
            boxes = np.asarray(out['bboxes'], dtype=np.float32) \
                if len(out['bboxes']) else np.zeros((0, 4), dtype=np.float32)
            labels = np.asarray(out['labels'], dtype=np.int64) \
                if len(out['labels']) else np.zeros((0,), dtype=np.int64)

        # HWC uint8 -> CHW float [0,1]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes_t,
            'labels': labels_t,
            'image_id': torch.tensor([idx]),
            'area': ((boxes_t[:, 2] - boxes_t[:, 0]) *
                     (boxes_t[:, 3] - boxes_t[:, 1])
                     if len(boxes_t) else torch.zeros((0,))),
            'iscrowd': torch.zeros((len(labels_t),), dtype=torch.int64),
        }
        return img_t, target


def detection_collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets


# ---------------- 亚组推断 + 索引生成 ----------------
def infer_subgroup(image_path: str) -> str:
    """文件名以 P 开头 -> 上颌；M 开头 -> 下颌。"""
    stem = os.path.splitext(os.path.basename(image_path))[0].upper()
    if stem.startswith('P'):
        return 'maxilla'
    if stem.startswith('M'):
        return 'mandible'
    return 'unknown'


def build_index_dataframe(voc_root: str) -> pd.DataFrame:
    """
    VOC 目录：
        voc_root/JPEGImages/*.jpg|png
        voc_root/Annotations/*.xml
    """
    jpeg_dir = os.path.join(voc_root, 'JPEGImages')
    ann_dir = os.path.join(voc_root, 'Annotations')
    rows = []
    for f in sorted(os.listdir(jpeg_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        stem = os.path.splitext(f)[0]
        xml = os.path.join(ann_dir, stem + '.xml')
        if not os.path.exists(xml):
            continue
        boxes, labels = VOCDetectionDataset._parse_xml(xml)
        if len(labels) == 0:
            continue
        label = 'fill' if labels[0] == 1 else 'rct'
        rows.append({
            'image_path': os.path.join(jpeg_dir, f),
            'xml_path': xml,
            'label': label,
            'subgroup': infer_subgroup(f),
        })
    return pd.DataFrame(rows)
