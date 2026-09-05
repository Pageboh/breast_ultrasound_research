import os
import glob
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import torch.nn.functional as F

class BUSIClassificationDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, path

def get_data_splits(data_dir, test_size=0.2, seed=42):
    file_paths = []
    labels = []
    classes = {'benign': 0, 'malignant': 1}
    
    for cls_name, label in classes.items():
        cls_folder = os.path.join(data_dir, cls_name)
        all_files = glob.glob(os.path.join(cls_folder, "*.png"))
        orig_images = [f for f in all_files if "_mask" not in os.path.basename(f)]
        for img_path in orig_images:
            file_paths.append(img_path)
            labels.append(label)

    _, val_files, _, val_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return val_files, val_labels

def main():
    parser = argparse.ArgumentParser()
    # 默认路径已指向你的 data 目录
    parser.add_argument('--data_dir', type=str, default='./data/BUSI', help='Path to BUSI dataset')
    parser.add_argument('--checkpoints_dir', type=str, default='./code3/checkpoints')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_f, val_l = get_data_splits(args.data_dir)
    val_loader = DataLoader(BUSIClassificationDataset(val_f, val_l, val_transform), 
                            batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 1. 初始化并加载 Swin-Tiny
    swin_model = timm.create_model('swin_tiny_patch4_window7_224', num_classes=2)
    swin_weight_path = os.path.join(args.checkpoints_dir, 'swin_tiny_best.pth')
    swin_model.load_state_dict(torch.load(swin_weight_path, map_location=device))
    swin_model = swin_model.to(device)
    swin_model.eval()

    # 2. 初始化并加载 ConvNeXt-Tiny
    cnn_model = timm.create_model('convnext_tiny', num_classes=2)
    cnn_weight_path = os.path.join(args.checkpoints_dir, 'convnext_tiny_best.pth')
    cnn_model.load_state_dict(torch.load(cnn_weight_path, map_location=device))
    cnn_model = cnn_model.to(device)
    cnn_model.eval()

    all_labels, all_paths = [], []
    swin_preds, swin_probs = [], []
    cnn_preds, cnn_probs = [], []

    print("--- 正在评估模型并提取预测结果 ---")
    with torch.no_grad():
        for imgs, labels, paths in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

            # Swin 预测
            s_out = swin_model(imgs)
            s_prob = F.softmax(s_out, dim=1)[:, 1].cpu().numpy()
            s_pred = s_out.argmax(dim=1).cpu().numpy()
            swin_probs.extend(s_prob)
            swin_preds.extend(s_pred)

            # ConvNeXt 预测
            c_out = cnn_model(imgs)
            c_prob = F.softmax(c_out, dim=1)[:, 1].cpu().numpy()
            c_pred = c_out.argmax(dim=1).cpu().numpy()
            cnn_probs.extend(c_prob)
            cnn_preds.extend(c_pred)

    # 计算全局指标
    swin_acc = accuracy_score(all_labels, swin_preds)
    swin_auc = roc_auc_score(all_labels, swin_probs)
    cnn_acc = accuracy_score(all_labels, cnn_preds)
    cnn_auc = roc_auc_score(all_labels, cnn_probs)

    print("\n--- 任务一模型对比结果 ---")
    print(f"[Swin-Tiny]    Accuracy: {swin_acc:.4f} | AUC: {swin_auc:.4f}")
    print(f"[ConvNeXt-Tiny] Accuracy: {cnn_acc:.4f} | AUC: {cnn_auc:.4f}")

    # 提取并分类 Bad Cases
    df = pd.DataFrame({
        'image_path': all_paths,
        'true_label': all_labels,
        'swin_pred': swin_preds,
        'cnn_pred': cnn_preds
    })
    
    # 映射标签名称以便于阅读
    label_map = {0: 'benign', 1: 'malignant'}
    df['true_label'] = df['true_label'].map(label_map)
    df['swin_pred'] = df['swin_pred'].map(label_map)
    df['cnn_pred'] = df['cnn_pred'].map(label_map)

    # 筛选分歧样本
    swin_right_cnn_wrong = df[(df['swin_pred'] == df['true_label']) & (df['cnn_pred'] != df['true_label'])]
    cnn_right_swin_wrong = df[(df['cnn_pred'] == df['true_label']) & (df['swin_pred'] != df['true_label'])]
    both_wrong = df[(df['swin_pred'] != df['true_label']) & (df['cnn_pred'] != df['true_label'])]

    bad_cases_path = os.path.join(args.checkpoints_dir, 'bad_cases_analysis.csv')
    df.to_csv(bad_cases_path, index=False)

    print("\n--- 差异性分析 (Bad Cases) ---")
    print(f"Swin 预测正确，但 ConvNeXt 预测错误: {len(swin_right_cnn_wrong)} 例")
    print(f"ConvNeXt 预测正确，但 Swin 预测错误: {len(cnn_right_swin_wrong)} 例")
    print(f"两路模型均预测错误: {len(both_wrong)} 例")
    print(f"详细预测对比日志已保存至: {bad_cases_path}")

if __name__ == '__main__':
    main()