import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# ==========================================
# 1. 自动寻路 & 数据集加载模块 (Dataset)
# ==========================================
class BUSIDataset(Dataset):
    def __init__(self, base_data_dir, img_size=256):
        self.image_paths = []
        self.mask_paths = []
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        # 智能寻路：自动在文件夹及其子文件夹中寻找包含 benign 和 malignant 的那一层
        true_root = base_data_dir
        for root, dirs, files in os.walk(base_data_dir):
            dirs_lower = [d.lower() for d in dirs]
            if 'benign' in dirs_lower and 'malignant' in dirs_lower:
                true_root = root
                break
                
        print(f"📂 自动定位到数据集真实根目录: {true_root}")

        classes = ['benign', 'malignant']
        for cls in classes:
            cls_dirs = [d for d in os.listdir(true_root) if d.lower() == cls]
            if not cls_dirs: continue
            cls_dir = os.path.join(true_root, cls_dirs[0])
            
            # 找到所有的原始图像
            all_files = os.listdir(cls_dir)
            img_files = [f for f in all_files if 'mask' not in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in img_files:
                img_path = os.path.join(cls_dir, img_file)
                base_name, ext = os.path.splitext(img_file)
                
                # 寻找对应的 mask 文件 (兼容原后缀或固定.png后缀)
                mask_path_1 = os.path.join(cls_dir, f"{base_name}_mask{ext}")
                mask_path_2 = os.path.join(cls_dir, f"{base_name}_mask.png")
                
                if os.path.exists(mask_path_1):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path_1)
                elif os.path.exists(mask_path_2):
                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path_2)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # 灰度图
        
        img = self.transform(img)
        mask = self.transform(mask)
        
        # 二值化
        mask = (mask > 0).float()
        
        filename = os.path.basename(img_path)
        return img, mask, filename


# ==========================================
# 2. U-Net 模型结构模块
# ==========================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        return torch.sigmoid(self.outc(x))


# ==========================================
# 3. 评估与可视化函数
# ==========================================
def calc_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    dice = (2. * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8)
    iou = (intersection + 1e-8) / (union + 1e-8)
    return dice.item(), iou.item()

def save_overlay(img_tensor, gt_tensor, pred_tensor, save_path):
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    
    gt = gt_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().cpu().numpy()
    pred = (pred > 0.5).astype(np.uint8)
    
    overlay = img.copy()
    overlay[gt == 1] = [0, 255, 0]             # GT 为绿色
    overlay[pred == 1] = [255, 0, 0]           # 预测 为红色
    overlay[(gt == 1) & (pred == 1)] = [255, 255, 0] # 重合 为黄色
    
    blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))


# ==========================================
# 4. 主流程 (训练与产出)
# ==========================================
def main():
    # 路径设置
    data_dir = "/root/autodl-tmp/breast_ultrasound_research/data" 
    result_dir = "/root/autodl-tmp/breast_ultrasound_research/code2/result"
    overlay_dir = os.path.join(result_dir, "mask_overlay_examples")
    
    # 创建结果文件夹
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    # 检查数据集
    dataset = BUSIDataset(data_dir)
    print(f"📊 成功读取并配对数据集，共提取 {len(dataset)} 个有效样本！")
    
    if len(dataset) == 0:
        print("❌ 错误：样本数为 0，请确认 /data/ 文件夹内是否真的有解压好的图片文件！")
        return

    # 划分训练集和测试集
    batch_size = 8
    epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 初始化模型
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    print(f"🚀 开始在 {device} 上训练...")
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
        
        print(f"   Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}")

    print("\n✅ 训练完成！正在生成测试集产出文件...")
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, (imgs, masks, filenames) in enumerate(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            
            dice, iou = calc_metrics(preds, masks)
            filename = filenames[0]
            
            results.append({
                "Model": "U-Net",
                "Filename": filename,
                "Dice": round(dice, 4),
                "IoU": round(iou, 4)
            })
            
            # 保存前 20 张验证集的可视化
            if i < 20:
                save_path = os.path.join(overlay_dir, f"pred_{filename}")
                save_overlay(imgs[0], masks[0], preds[0], save_path)

    # 汇总 CSV 指标
    csv_path = os.path.join(result_dir, "segmentation_metrics.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    print(f"\n🎉 运行成功结束！请在以下路径查看结果：")
    print(f"1. 评估指标表格: {csv_path} (平均Dice: {df['Dice'].mean():.4f})")
    print(f"2. 分割可视化图: {overlay_dir}/")

if __name__ == "__main__":
    main()