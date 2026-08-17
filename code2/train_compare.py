import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import random

# ==========================================
# 0. 统一训练设置：固定随机种子 (非常关键)
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 1. 智能寻路数据集 (保持阶段3A逻辑)
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, base_data_dir, img_size=256):
        self.image_paths = []
        self.mask_paths = []
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        true_root = base_data_dir
        for root, dirs, files in os.walk(base_data_dir):
            dirs_lower = [d.lower() for d in dirs]
            if 'benign' in dirs_lower and 'malignant' in dirs_lower:
                true_root = root
                break

        classes = ['benign', 'malignant']
        for cls in classes:
            cls_dirs = [d for d in os.listdir(true_root) if d.lower() == cls]
            if not cls_dirs: continue
            cls_dir = os.path.join(true_root, cls_dirs[0])
            
            img_files = [f for f in os.listdir(cls_dir) if 'mask' not in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in img_files:
                img_path = os.path.join(cls_dir, img_file)
                base_name, ext = os.path.splitext(img_file)
                mask_path_1 = os.path.join(cls_dir, f"{base_name}_mask{ext}")
                mask_path_2 = os.path.join(cls_dir, f"{base_name}_mask.png")
                
                if os.path.exists(mask_path_1):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path_1)
                elif os.path.exists(mask_path_2):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path_2)

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        mask = self.transform(Image.open(self.mask_paths[idx]).convert('L'))
        return img, (mask > 0).float(), os.path.basename(self.image_paths[idx])

# ==========================================
# 2. 模型结构定义 (基础组件)
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

# 结构 A：标准 U-Net
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.conv1(torch.cat([x3, self.up1(x4)], dim=1))
        x = self.conv2(torch.cat([x2, self.up2(x)], dim=1))
        x = self.conv3(torch.cat([x1, self.up3(x)], dim=1))
        return torch.sigmoid(self.outc(x))

# 结构 B：Attention Gate 与 Attention U-Net
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1), nn.BatchNorm2d(1), nn.Sigmoid())
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.relu(g1 + x1)
        return x * self.psi(psi)

class AttUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv1 = DoubleConv(3, 64)
        self.Conv2 = DoubleConv(64, 128)
        self.Conv3 = DoubleConv(128, 256)
        self.Conv4 = DoubleConv(256, 512)

        self.Up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.Att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.Up_conv3 = DoubleConv(512, 256)

        self.Up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.Att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.Up_conv2 = DoubleConv(256, 128)

        self.Up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.Att1 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.Up_conv1 = DoubleConv(128, 64)

        self.Conv_1x1 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Conv2(self.Maxpool(e1))
        e3 = self.Conv3(self.Maxpool(e2))
        e4 = self.Conv4(self.Maxpool(e3))

        d3 = self.Up_conv3(torch.cat((self.Att3(g=self.Up3(e4), x=e3), self.Up3(e4)), dim=1))
        d2 = self.Up_conv2(torch.cat((self.Att2(g=self.Up2(d3), x=e2), self.Up2(d3)), dim=1))
        d1 = self.Up_conv1(torch.cat((self.Att1(g=self.Up1(d2), x=e1), self.Up1(d2)), dim=1))

        return torch.sigmoid(self.Conv_1x1(d1))

# 结构 C：U-Net++ (Nested U-Net，深度调整为3以对齐参数量)
class UNetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(3, 64)
        self.conv1_0 = DoubleConv(64, 128)
        self.conv2_0 = DoubleConv(128, 256)
        self.conv3_0 = DoubleConv(256, 512)

        self.conv0_1 = DoubleConv(64+128, 64)
        self.conv1_1 = DoubleConv(128+256, 128)
        self.conv2_1 = DoubleConv(256+512, 256)

        self.conv0_2 = DoubleConv(64*2+128, 64)
        self.conv1_2 = DoubleConv(128*2+256, 128)

        self.conv0_3 = DoubleConv(64*3+128, 64)
        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        return torch.sigmoid(self.final(x0_3))

# ==========================================
# 3. 训练与对比汇总逻辑
# ==========================================
def calc_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    dice = (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)
    iou = (intersection + 1e-8) / (union + 1e-8)
    return dice.item(), iou.item()

def main():
    data_dir = "/root/autodl-tmp/breast_ultrasound_research/data" 
    result_dir = "/root/autodl-tmp/breast_ultrasound_research/code2/result"
    os.makedirs(result_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = BUSIDataset(data_dir)
    print(f"📊 数据集加载成功: {len(dataset)} 个样本")

    # 固定的数据集切分
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    models_dict = {
        "UNet": UNet().to(device),
        "AttUNet": AttUNet().to(device),
        "UNetPlusPlus": UNetPlusPlus().to(device)
    }

    # 用于汇总两张表的数据结构
    detailed_results = {} # { filename: { "UNet_Dice": x, "AttUNet_Dice": y ... } }
    summary_results = []  # [ {"Model": "UNet", "Mean_Dice": x, "Mean_IoU": y} ... ]

    epochs = 20 # 统一的训练轮次
    
    for model_name, model in models_dict.items():
        print(f"\n🚀 开始统一标准训练模型: {model_name}")
        criterion = nn.BCELoss() 
        optimizer = optim.Adam(model.parameters(), lr=1e-4) # 统一的优化器和学习率
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for imgs, masks, _ in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            print(f"   Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}")
        
        # 评测当前模型
        print(f"🔍 评测 {model_name} 中...")
        model.eval()
        model_dice_sum, model_iou_sum = 0, 0
        
        with torch.no_grad():
            for imgs, masks, filenames in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                dice, iou = calc_metrics(preds, masks)
                filename = filenames[0]
                
                # 记录单个图像结果 (给 dice_iou_table.csv 用)
                if filename not in detailed_results:
                    detailed_results[filename] = {"Filename": filename}
                detailed_results[filename][f"{model_name}_Dice"] = round(dice, 4)
                detailed_results[filename][f"{model_name}_IoU"] = round(iou, 4)
                
                model_dice_sum += dice
                model_iou_sum += iou
        
        # 记录模型平均指标 (给 seg_model_comparison.csv 用)
        summary_results.append({
            "Model": model_name,
            "Mean_Dice": round(model_dice_sum / test_size, 4),
            "Mean_IoU": round(model_iou_sum / test_size, 4)
        })

    # ==========================================
    # 4. 产出导出
    # ==========================================
    # 产出 1: seg_model_comparison.csv (宏观对比)
    df_summary = pd.DataFrame(summary_results)
    df_summary.to_csv(os.path.join(result_dir, "seg_model_comparison.csv"), index=False)
    
    # 产出 2: dice_iou_table.csv (微观细粒度对比)
    df_details = pd.DataFrame(detailed_results.values())
    df_details.to_csv(os.path.join(result_dir, "dice_iou_table.csv"), index=False)

    print("\n🎉 全部模型对比实验完成！")
    print(f"✅ 宏观对比表已保存至: result/seg_model_comparison.csv")
    print(f"✅ 微观统计表已保存至: result/dice_iou_table.csv")
    print("\n宏观对比预览:\n", df_summary.to_string(index=False))

if __name__ == "__main__":
    main()