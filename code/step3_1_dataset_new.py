import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os

# 1. 自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, df,image_columns,labels, transform=None):
        self.df = df.copy()
        self.transform = transform
        
        # 提取特征和标签
        self.labels = torch.tensor(self.df[labels].values, dtype=torch.long)
       
        # 图像路径列
        self.image_columns = image_columns
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        # 获取图像
        images = []
        for col in self.image_columns:
            img_path = self.df[col].iloc[idx]
            if pd.isna(img_path) or not os.path.exists(img_path):
                # 如果图片不存在,创建空白图片
                img = Image.new('RGB', (224, 224), color='black')
            else:
                img = Image.open(img_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
            
        # 将图像堆叠为一个tensor
        images = torch.stack(images)
        
        return {
            'images': images,
            'label': self.labels[idx]
        }
