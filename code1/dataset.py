import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class BUSIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 设定标签映射 (0: 良性, 1: 恶性, 2: 正常)
        class_map = {'benign': 0, 'malignant': 1, 'normal': 2}
        
        for class_name, label in class_map.items():
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 找不到文件夹 {class_dir}")
                continue
                
            for img_name in os.listdir(class_dir):
                # 核心逻辑：过滤掉所有带有 'mask' 字样的标注图，只读取原图
                if 'mask' not in img_name.lower(): 
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 必须转换为 RGB，因为预训练模型是基于 3 通道彩色图训练的
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

def get_transform():
    # 深度学习标准预处理套件
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 自动测试模块
if __name__ == '__main__':
    data_dir = '/root/autodl-tmp/breast_ultrasound_research/data/BUSI'
    dataset = BUSIDataset(data_dir, transform=get_transform())
    
    print("="*40)
    print(f"成功加载数据集！")
    print(f"共找到 {len(dataset)} 张有效的超声原图（已自动剔除 Mask）。")
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"单张图像的 Tensor 形状已统一为: {img.shape}")
        print(f"第一张图像的对应标签为: {label}")
    print("="*40)