import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torchvision import transforms
from dataset import BUSIDataset
from model_factory import get_model

# 1. 基础设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
models_to_eval = ['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2', 'efficientnet_b0']

# 更新为你提供的数据准确路径
data_path = '/root/autodl-tmp/breast_ultrasound_research/data/BUSI'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
checkpoint_dir = os.path.join(project_root, 'checkpoints')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

torch.manual_seed(42)
full_dataset = BUSIDataset(data_path, transform=transform) 
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = random_split(full_dataset, [train_size, val_size])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

results = []
wrong_cases_list = []

# 2. 循环评估
for name in models_to_eval:
    print(f"\n正在评估模型: {name} ...")
    model = get_model(name, num_classes=2).to(device)
    weight_path = os.path.join(checkpoint_dir, f'{name}_best.pth')
    
    if not os.path.exists(weight_path):
        print(f"⚠️ 未找到 {name} 的权重文件，跳过。")
        continue
        
    print(f"✅ 成功读取权重: {weight_path}")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    actual_idx = idx * batch_size + i
                    img_path = full_dataset.image_paths[val_dataset.indices[actual_idx]]
                    wrong_cases_list.append({
                        'model': name,
                        'image_path': img_path,
                        'true_label': labels[i].item(),
                        'pred_label': preds[i].item(),
                        'malignant_prob': probs[i].item()
                    })

    # 计算指标
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results.append({
        'Model': name,
        'Accuracy': f"{acc:.4f}",
        'AUC': f"{auc:.4f}",
        'Sensitivity': f"{sensitivity:.4f}",
        'Specificity': f"{specificity:.4f}"
    })

print("\n" + "="*40)
df_metrics = pd.DataFrame(results)
df_metrics.to_csv('cnn_model_comparison.csv', index=False)
print("=== 模型对比基线已生成 (cnn_model_comparison.csv) ===")
print(df_metrics)

df_wrong = pd.DataFrame(wrong_cases_list)
df_wrong.to_csv('wrong_cases.csv', index=False)
print(f"=== 错误病例列表已导出 (wrong_cases.csv) ===")
print("="*40)