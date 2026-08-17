import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from dataset import BUSIDataset
from model_factory import get_model

# 1. 基础设置与准确路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 10
models_to_train = ['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2', 'efficientnet_b0']

# 数据集的准确绝对路径
data_path = '/root/autodl-tmp/breast_ultrasound_research/data/BUSI'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(project_root, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# 创建用于存放图片输出的目录
output_dir = os.path.join(project_root, 'output')
os.makedirs(output_dir, exist_ok=True)

# 2. 数据准备
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torch.manual_seed(42)
full_dataset = BUSIDataset(data_path, transform=transform)

if len(full_dataset) == 0:
    raise ValueError(f"在 {data_path} 下没有找到图片，请检查路径！")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 3. 循环训练所有模型
for model_name in models_to_train:
    print(f"\n==================== 正在启动 {model_name} 训练 ====================")
    model = get_model(model_name, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_acc = 0.0
    
    # 记录每个epoch的数据用于画图
    history_train_loss = []
    history_val_auc = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        epoch_train_loss = running_loss / len(train_loader)
        history_train_loss.append(epoch_train_loss)
            
        # 验证逻辑与AUC计算
        model.eval()
        correct = 0
        total = 0
        
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                # 获取预测概率 (应用 softmax)
                probs = F.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 收集用于计算 AUC 的真实标签和预测概率 (正类的概率)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy()) 

        val_acc = 100 * correct / total
        
        # 计算当前 epoch 的 AUC
        try:
            val_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            # 防止验证集类别不全导致的报错
            val_auc = 0.0
            print("Warning: Only one class present in y_true. ROC AUC score is not defined in that case.")
            
        history_val_auc.append(val_auc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_train_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        
        # 保存最佳权重
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"{model_name} 训练完成，最佳权重已保存至 {checkpoint_dir}")
    
    # 绘制并保存曲线图
    epochs_range = range(1, num_epochs + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 画 Training Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_train_loss, label='Train Loss', color='blue')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 画 Validation AUC 曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_val_auc, label='Validation AUC', color='green')
    plt.title(f'{model_name} - Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # 调整布局并保存
    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, f'{model_name}_loss_auc_curves.png')
    plt.savefig(plot_save_path)
    plt.close()
    
    print(f"{model_name} 训练曲线已保存至 {plot_save_path}")