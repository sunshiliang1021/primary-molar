import torch
import torch.nn as nn
import torchvision.models as models

# 2. 多模态模型 - 更新处理10张图片 修改 MultiModalNet 类中的维度计算

class MultiModalNet_ResNet18(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(512, 128)  # 保持这个维度不变
        
  
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features  # 1张图片，每张128维特征
      
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features, 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历1张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 11*128]
    
    
        output = self.fusion(image_features)
        
        return output
    
class MultiModalNet_ResNet50(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(2048, 128)  # 保持这个维度不变
        
  
        
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features  # 应该是5，而不是11
        
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features , 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历10张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 10*128]
        
     
        output = self.fusion(image_features)
        
        return output
    
class MultiModalNet_ResNet101(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.cnn.fc = nn.Linear(2048, 128)  # 保持这个维度不变
        
    
        
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features # 10张图片，每张128维特征
       
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features , 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历10张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 10*128]
        
        output = self.fusion(image_features)
        
        return output
    
class MultiModalNet_AlexNet(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        self.cnn.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128)
        )
        
        
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features  # 10张图片，每张128维特征
       
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features, 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历10张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 10*128]
        
       
         
        output = self.fusion(image_features)
        
        return output
    
class MultiModalNet_VGG16(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.vgg16(pretrained=True)
        self.cnn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 128)
        )
        
     
        
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features  # 10张图片，每张128维特征
        
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features , 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历10张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 10*128]
        
        output = self.fusion(image_features)
        
        return output
    
class MultiModalNet_Inception3(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.inception_v3(pretrained=True)
        self.cnn.fc = nn.Linear(2048, 128)  # 保持这个维度不变
        
        
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features # 10张图片，每张128维特征
        
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features, 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历10张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 10*128]
        
        output = self.fusion(image_features)
        
        return output
    
class MultiModalNet_GoogLeNet(nn.Module):
    def __init__(self, num_image_features):
        super().__init__()
        
        # 图像特征提取
        self.cnn = models.googlenet(pretrained=True)
        self.cnn.fc = nn.Linear(1024, 128)  # 保持这个维度不变
        
        
        # 特征融合 - 修正维度计算
        total_image_features = 128 * num_image_features  # 10张图片，每张128维特征
     
        
        self.fusion = nn.Sequential(
            nn.Linear(total_image_features, 512),  # 修改输入维度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        
    def forward(self, images):
        batch_size = images.size(0)
        
        # 处理每个图像
        image_features = []
        for i in range(images.size(1)):  # 遍历10张图片
            img_feature = self.cnn(images[:, i])
            image_features.append(img_feature)
        
        # 合并图像特征
        image_features = torch.cat(image_features, dim=1)  # [batch_size, 10*128]
        
      
        
        output = self.fusion(image_features)
        
        return output