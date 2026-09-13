import os
import torch
import torch.nn as nn
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
    ssd300_vgg16,
    SSD300_VGG16_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

NUM_CLASSES = 3   # 0=background, 1=fill, 2=rct


def build_faster_rcnn(num_classes=NUM_CLASSES, pretrained=True):
    """两阶段检测器：Faster R-CNN (ResNet50-FPN)，ImageNet/COCO 预训练迁移。"""
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def build_ssd(num_classes=NUM_CLASSES, pretrained=True):
    """单阶段检测器：SSD300-VGG16。"""
    weights = SSD300_VGG16_Weights.DEFAULT if pretrained else None
    model = ssd300_vgg16(weights=weights)
    num_anchors = model.head.classification_head.num_anchors
    in_channels = model.head.classification_head.conv[0][0].in_channels
    model.head.classification_head.num_classes = num_classes
    cls_logits = nn.Conv2d(in_channels, num_anchors * num_classes,
                           kernel_size=3, padding=1)
    nn.init.normal_(cls_logits.weight, std=0.01)
    nn.init.constant_(cls_logits.bias, 0)
    model.head.classification_head.cls_logits = cls_logits
    return model


# ---------- YOLOv7-X 接口占位（真实训练走 subprocess） ----------
def yolov7x_weights_path(repo_root: str) -> str:
    """官方预训练权重路径（ImageNet 预训练 -> 迁移学习）。"""
    return os.path.join(repo_root, 'yolov7x.pt')


def yolov7x_dir(repo_root: str) -> str:
    return os.path.join(repo_root, 'cfg', 'training', 'yolov7x.yaml')
