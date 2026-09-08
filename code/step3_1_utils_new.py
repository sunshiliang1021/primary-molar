import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import platform
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve, average_precision_score
)
import torch
#plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def plot_confusion_matrix(y_true, y_pred, save_path):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()

def plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir, model_name):
    """
    绘制训练过程中的损失和准确率曲线(左右布局)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  # 改为1行2列
    
    # 绘制损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)  # 添加网格线，使曲线更容易读取
    # 设置横坐标为整数
    num_epochs = len(train_losses)
    ax1.set_xticks(range(num_epochs))
    ax1.set_xticklabels(range(1, num_epochs + 1))

    # 绘制准确率曲线
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    #ax2.set_ylim(0, 100)  # 设置y轴范围从0到100
    ax2.legend()
    ax2.grid(True)  # 添加网格线，使数值更容易读取
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_training_curves.png')
    plt.close()

def plot_roc_comparison(results, save_path):
    """
    绘制多个模型的ROC曲线比较
    """
    plt.figure(figsize=(10, 8))
    
    for result in results:
        fpr, tpr, _ = roc_curve(result['labels'], result['predictions'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 
                label=f"{result['model_name']} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def calculate_metrics(y_true, y_pred):
    """
    计算各种评估指标
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }


def plot_f1_curve(train_f1s, val_f1s, output_dir, model_name):
    """
    绘制F1分数曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_f1s, label='Train F1')
    plt.plot(val_f1s, label='Val F1')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_name}_f1_curves.png')
    plt.close()


def plot_single_roc(labels, predictions, save_path):
    """
    绘制单个模型的ROC曲线
    """
    plt.figure(figsize=(8, 6))
    
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc 



def plot_single_pr_curve(labels, predictions, save_path):
    """
    绘制单个模型的PR曲线
    """
    plt.figure(figsize=(8, 6))
    
    precision, recall, _ = precision_recall_curve(labels, predictions)
    ap = average_precision_score(labels, predictions)
    
    plt.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.ylim([0.0, 1.05])  # 设置y轴从0开始
    plt.xlim([0.0, 1.0])   # 设置x轴范围
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return ap

def plot_pr_comparison(results, save_path):
    """
    绘制多个模型的PR曲线比较
    """
    plt.figure(figsize=(10, 8))
    
    for result in results:
        precision, recall, _ = precision_recall_curve(result['labels'], result['predictions'])
        ap = average_precision_score(result['labels'], result['predictions'])
        plt.plot(recall, precision, 
                label=f"{result['model_name']} (AP = {ap:.3f})")
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.ylim([0.0, 1.05])  # 设置y轴从0开始
    plt.xlim([0.0, 1.0])   # 设置x轴范围
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 

def train_epoch(model, train_loader, criterion, optimizer, device, max_grad_norm=0.5):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        numerical = batch['numerical'].to(device)
        images = batch['images'].to(device)
        labels = batch['label'].to(device)
        
        # 检查并跳过无效batch
        if numerical.size(0) < 2:
            continue
            
        optimizer.zero_grad()
        
        try:
            outputs = model(images, numerical)
            loss = criterion(outputs, labels)
            
            # 检查loss是否为NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected")
                continue
                
            loss.backward()
            
            # 检查梯度是否为NaN
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print("Warning: NaN gradient detected")
                    continue
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
        except RuntimeError as e:
            print(f"Error during training: {e}")
            continue
    
    # 确保有足够的样本进行评估
    if len(all_preds) == 0:
        return float('inf'), {'Accuracy': 0, 'Precision': 0, 'Recall': 0, 'F1': 0}
        
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return total_loss / len(train_loader), metrics 