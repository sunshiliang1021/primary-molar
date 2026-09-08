import os
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from torchvision import transforms
from sklearn.model_selection import train_test_split
from step3_1_models_new import MultiModalNet_ResNet18 ,MultiModalNet_ResNet50,MultiModalNet_ResNet101,MultiModalNet_AlexNet,MultiModalNet_VGG16,MultiModalNet_GoogLeNet
from step3_1_dataset_new import MultiModalDataset
from step3_1_utils_new import (
    plot_metrics, 
    plot_confusion_matrix, 
    plot_roc_comparison, 
    calculate_metrics,
    plot_single_roc,
    plot_single_pr_curve,
    plot_pr_comparison,
   
)
from torch.utils.data import WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import gc

def train_epoch(model, train_loader, criterion, optimizer, device,scaler):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    #all_probs = []  # 存储概率值
    
    for batch in train_loader:
        images = batch['images'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()

        with autocast():
            outputs = model(images) 
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        predicted = (probs > 0.5).float()  # 将概率转换为二分类预测
        
        #all_probs.extend(probs.cpu().numpy())  # 保存概率值
        all_preds.extend(predicted.cpu().numpy())  # 保存二分类预测
        all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    #all_probs = np.array(all_probs)
    
    # 使用二分类预测计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    return total_loss / len(train_loader), metrics

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 存储概率值
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probs > 0.5).float()  # 将概率转换为二分类预测
            
            all_probs.extend(probs.cpu().numpy())  # 保存概率值
            all_preds.extend(predicted.cpu().numpy())  # 保存二分类预测
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 使用二分类预测计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 返回概率值用于ROC和PR曲线
    return total_loss / len(val_loader), metrics, all_probs, all_labels
def test(model, test_loader, criterion, device, test_df, model_name, output_dir):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []  # 存储概率值
    
    with torch.no_grad():
        for batch in test_loader:
          
            images = batch['images'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)[:, 1]
            predicted = (probs > 0.5).float()  # 将概率转换为二分类预测
            
            all_probs.extend(probs.cpu().numpy())  # 保存概率值
            all_preds.extend(predicted.cpu().numpy())  # 保存二分类预测
            all_labels.extend(labels.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 使用二分类预测计算指标
    metrics = calculate_metrics(all_labels, all_preds)
    
    # 将预测结果添加到测试数据中
    test_df[f'{model_name}_pred'] = all_preds
    #test_df[f'{model_name}_prob'] = all_probs
    
    # 保存带预测结果的测试数据
    result_path = os.path.join(output_dir, f'{model_name}_test_predictions.xlsx')
    test_df.to_excel(result_path, index=False)
    print(f"保存测试预测结果到: {result_path}")
    
    return total_loss / len(test_loader), metrics, all_probs, all_labels
def create_output_dir(base_path):
    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_path, f'output/results_{current_time}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def main():
    
    
    # 设置随机种子和设备
    torch.manual_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 定义特征名称
    labels = 'label'
    image_names = ['picture']
    
    num_image_features=len(image_names)
    batch_size = 4
    num_epochs = 20
    early_stop_patience = 6 #早停次数5
    scheduler_patience = 3 #3
    
    # 加载数据
    excel_path = r'/root/workspace/tooth/data/image_paths2.csv'
    df = pd.read_csv(excel_path)
    base_path = os.path.dirname(os.path.dirname(excel_path))  # 去掉 /data/image_paths.csv，获取项目根目录
    output_dir = create_output_dir(base_path)

    # 数据预处理
    #由于是医学人脸图像，建议不需要色彩扰动，随机剪裁了
    '''
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪增强
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 色彩扰动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])
    '''
    train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),  # 轻微平移/旋转
    transforms.ColorJitter(brightness=0.1),     # 轻微亮度变化（模拟不同设备）
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])


    # 划分数据集
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42,stratify=df[labels])
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42,stratify=train_val_df[labels])
      
    # 创建数据加载器
    train_dataset = MultiModalDataset(train_df, image_columns=image_names,labels=labels, transform=train_transform)
    val_dataset = MultiModalDataset(val_df, image_columns=image_names,labels=labels, transform=val_test_transform)
    test_dataset = MultiModalDataset(test_df,image_columns=image_names,labels=labels,  transform=val_test_transform)
   
    scaler = GradScaler()
    # 创建模型
    models = {
        'MultiModalNet_ResNet18': MultiModalNet_ResNet18(num_image_features=num_image_features),
        'MultiModalNet_ResNet50': MultiModalNet_ResNet50(num_image_features=num_image_features),
        'MultiModalNet_ResNet101': MultiModalNet_ResNet101(num_image_features=num_image_features),
        'MultiModalNet_AlexNet': MultiModalNet_AlexNet(num_image_features=num_image_features), 
        'MultiModalNet_VGG16': MultiModalNet_VGG16(num_image_features=num_image_features),
        'MultiModalNet_GoogLeNet': MultiModalNet_GoogLeNet(num_image_features=num_image_features)
    }
    
    # 存储所有模型的评估结果
    all_results = []
    
    
    for model_name, model in models.items():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f'\n训练 {model_name}...')
        # 自适应 lr 和 batch size
    
        if 'MultiModalNet_ResNet18' in model_name:
            batch_size = 4
            lr = 0.0003
        elif 'MultiModalNet_ResNet50' in model_name or 'MultiModalNet_GoogLeNet' in model_name:
            batch_size = 4
            lr = 0.0005
        else:  # ResNet101 / VGG16 / AlexNet
            batch_size = 2
            lr = 0.0002
        
        #lr = 0.0003
        
        # 类别权重 + sampler
        class_weights = compute_class_weight('balanced', classes=np.array([0,1]), y=train_df[labels])
        sample_weights = train_df[labels].map({0: class_weights[0], 1: class_weights[1]}).values
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, pin_memory=False, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=False, num_workers=1)

        model = model.__class__(num_image_features=num_image_features)  # 新建模型
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=scheduler_patience, verbose=True,min_lr=1e-6)

        best_val_acc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        train_f1s, val_f1s = [], []
        epoch_metrics = []
        
        # 训练循环
        for epoch in range(num_epochs):
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler )
            val_loss, val_metrics, val_preds, val_labels= validate(model, val_loader, criterion, device)
            
            # 学习率调度
            scheduler.step(val_metrics['Accuracy'])

            # 记录损失和指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_metrics['Accuracy'] * 100)
            val_accs.append(val_metrics['Accuracy'] * 100)
            train_f1s.append(train_metrics['F1'])
            val_f1s.append(val_metrics['F1'])
            #scheduler.step()
            # 保存每个epoch的指标
            epoch_metrics.append({
                'Epoch': epoch + 1,
                'Train_Accuracy': train_metrics['Accuracy'],
                'Train_Precision': train_metrics['Precision'],
                'Train_Recall': train_metrics['Recall'],
                'Train_F1': train_metrics['F1'],
                'Val_Accuracy': val_metrics['Accuracy'],
                'Val_Precision': val_metrics['Precision'],
                'Val_Recall': val_metrics['Recall'],
                'Val_F1': val_metrics['F1']
            })
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_metrics["Accuracy"]:.2%} | Val Loss: {val_loss:.4f}, Val Acc: {val_metrics["Accuracy"]:.2%}')
                
            #早停机制
            if val_metrics['Accuracy'] > best_val_acc:
                best_val_acc = val_metrics['Accuracy']
                early_stop_patience_counter = 0
                torch.save(model.state_dict(), f'{output_dir}/best_{model_name}.pth')
            else:
                early_stop_patience_counter += 1
                if early_stop_patience_counter >= early_stop_patience:
                    print(f"早停在第{epoch+1}轮")
                    break
            torch.cuda.empty_cache()

          
        # 绘制训练曲线
        plot_metrics(train_losses, val_losses, train_accs, val_accs, output_dir, model_name)
        #plot_f1_curve(train_f1s, val_f1s, output_dir, model_name)
        
        # 保存训练过程中的指标
        pd.DataFrame(epoch_metrics).to_csv(
            f'{output_dir}/{model_name}_training_metrics.csv', 
            index=False
        )
        
        # 加载最佳模型进行测试
        best_model = model.__class__(num_image_features=num_image_features).to(device)
       
        best_model.load_state_dict(torch.load(f'{output_dir}/best_{model_name}.pth'))
        test_loss, test_metrics, test_probs, test_labels = test(
            best_model, test_loader, criterion, device, test_df, model_name, output_dir
        )
       
        # 计算二分类硬预测
        test_preds = (test_probs > 0.5).astype(int)
        
        # 绘制单个模型的ROC曲线（使用概率值）
        roc_auc = plot_single_roc(
            test_labels, 
            test_probs,  # 使用概率值
            f'{output_dir}/{model_name}_roc.png'
        )
        
        # 绘制单个模型的PR曲线（使用概率值）
        ap_score = plot_single_pr_curve(
            test_labels, 
            test_probs,  # 使用概率值
            f'{output_dir}/{model_name}_pr.png'
        )
        
        # 绘制混淆矩阵（使用二分类预测）
        plot_confusion_matrix(
            test_labels, 
            test_preds,  # 使用二分类预测
            f'{output_dir}/{model_name}_confusion_matrix.png'
        )
        
        # 保存评估结果
        result = {
            'model_name': model_name,
            'accuracy': test_metrics['Accuracy'],
            'precision': test_metrics['Precision'],
            'recall': test_metrics['Recall'],
            'f1': test_metrics['F1'],
            'auc': roc_auc,
            'ap': ap_score,
            'predictions': test_probs,  # 保存概率值用于后续绘图
            'labels': test_labels
        }
        all_results.append(result)
        
        # 打印测试集结果
        print(f'\n{model_name} 测试集结果:')
        print(f'Accuracy: {test_metrics["Accuracy"]:.4f}')
        print(f'Precision: {test_metrics["Precision"]:.4f}')
        print(f'Recall: {test_metrics["Recall"]:.4f}')
        print(f'F1 Score: {test_metrics["F1"]:.4f}')
        print(f'AUC: {roc_auc:.4f}')
        print(f'AP: {ap_score:.4f}')
        # 强制进行垃圾回收
        del model, train_loader, val_loader, test_loader
        gc.collect()
        torch.cuda.empty_cache()
    # 绘制ROC曲线比较
    plot_roc_comparison(all_results, f'{output_dir}/model_comparison_roc.png')
    
    # 绘制PR曲线比较
    plot_pr_comparison(all_results, f'{output_dir}/model_comparison_pr.png')
    
    # 保存评估指标到Excel
    metrics_df = pd.DataFrame([{
        'Model': r['model_name'],
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1': r['f1'],
        'AUC': r['auc'],
        'AP': r['ap']
    } for r in all_results])
    
    # 保存详细的评估指标
    metrics_df.to_csv(f'{output_dir}/model_comparison_metrics.csv', index=False)
    
    # 打印所有模型的比较结果
    print("\n模型性能比较:")
    for _, row in metrics_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"准确率: {row['Accuracy']:.2%}")
        print(f"精确率: {row['Precision']:.2%}")
        print(f"召回率: {row['Recall']:.2%}")
        print(f"F1分数: {row['F1']:.2%}")
        print(f"AUC: {row['AUC']:.3f}")
        print(f"AP: {row['AP']:.3f}")
if __name__ == '__main__':
    main() 