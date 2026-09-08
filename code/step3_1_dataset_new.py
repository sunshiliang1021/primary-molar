import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

# 1. 自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, df, image_col, box_cols, label_col, transform=None):
        """
        参数：
            df: 包含标注信息的 DataFrame
            image_col: 图片路径列名，如 'image_path'
            box_cols: 边界框坐标列名列表，如 ['x1', 'y1', 'x2', 'y2']
            label_col: 目标类别列名，如 'class_id'
            transform: 图像预处理/增强
        """
        self.df = df.copy()
        self.transform = transform
        self.image_col = image_col
        self.box_cols = box_cols
        self.label_col = label_col
       
        # 图像路径列
        self.image_paths = self.df[image_col].unique()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        # 获取图像
            img_path = self.image_paths[idx]
            if pd.isna(img_path) or not os.path.exists(img_path):
                # 如果图片不存在,创建空白图片
                img = Image.new('RGB', (224, 224), color='black')
            else:
                img = Image.open(img_path).convert('RGB')
        
        records = self.df[self.df[self.image_col] == img_path]
        
        boxes = []
        labels = []
        for _, row in records.iterrows():
            box = row[self.box_cols].values.astype(float)
            label = int(row[self.label_col])
            boxes.append(box)
            labels.append(label)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            img = self.transform(img)
            images.append(img)

        target = {
            'boxes': boxes,           # shape: (N, 4)
            'labels': labels,         # shape: (N,)
            'image_id': torch.tensor([idx])
        }
        
        return img, target
       
def collate_fn(batch):
        """
        batch: list of tuples [(img1, target1), (img2, target2), ...]
        返回: (images_tuple, targets_list)
        """
        return tuple(zip(*batch))
