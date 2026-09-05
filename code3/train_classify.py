import os
import glob
import copy
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

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
    
    # 0: benign, 1: malignant
    classes = {'benign': 0, 'malignant': 1}
    for cls_name, label in classes.items():
        cls_folder = os.path.join(data_dir, cls_name)
        all_files = glob.glob(os.path.join(cls_folder, "*.png"))
        # 排除包含 mask 的标注文件
        orig_images = [f for f in all_files if "_mask" not in os.path.basename(f)]
        for img_path in orig_images:
            file_paths.append(img_path)
            labels.append(label)

    train_files, val_files, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    return train_files, val_files, train_labels, val_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/BUSI', help='Path to BUSI dataset')
    parser.add_argument('--model_name', type=str, required=True, choices=['swin_tiny', 'convnext_tiny'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理与增强 (224x224 分辨率)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_f, val_f, train_l, val_l = get_data_splits(args.data_dir)
    train_loader = DataLoader(BUSIClassificationDataset(train_f, train_l, train_transform), 
                              batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(BUSIClassificationDataset(val_f, val_l, val_transform), 
                            batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 模型定义 (映射 timm 预训练权重名称)
    timm_name = 'swin_tiny_patch4_window7_224' if args.model_name == 'swin_tiny' else 'convnext_tiny'
    model = timm.create_model(timm_name, pretrained=True, num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_acc = 0.0
    best_weights = None
    history = []

    print(f"--- Start training {args.model_name} on {device} ---")
    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()

        train_loss = train_loss / len(train_f)
        train_acc = train_correct / len(train_f)

        # 验证循环
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()

        val_loss = val_loss / len(val_f)
        val_acc = val_correct / len(val_f)

        history.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(args.output_dir, f"{args.model_name}_best.pth"))

        print(f"Epoch [{epoch+1:02d}/{args.epochs}] | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # 保存日志
    pd.DataFrame(history).to_csv(os.path.join(args.output_dir, f"{args.model_name}_history.csv"), index=False)
    print(f"Finished. Best Val Acc: {best_acc:.4f}")

if __name__ == '__main__':
    main()