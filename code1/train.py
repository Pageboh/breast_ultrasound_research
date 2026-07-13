import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
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
            
        # 极简验证逻辑
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")
        
        # 保存最佳权重
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            
    print(f"{model_name} 训练完成，最佳权重已保存至 {checkpoint_dir}")