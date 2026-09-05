import os
import glob
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# 1. 核心研究与分析底表构建
# [修改点] 使用服务器上的绝对路径，避免因为终端当前所在目录不同导致路径解析错误
data_dir = '/root/autodl-tmp/breast_ultrasound_research/data/BUSI' 
categories = ['benign', 'malignant'] 
records = []

for cat in categories:
    all_files = glob.glob(os.path.join(data_dir, cat, '*.png'))
    img_paths = [f for f in all_files if 'mask' not in f]
    
    for img_path in img_paths:
        mask_path = img_path.replace('.png', '_mask.png')
        if os.path.exists(mask_path):
            records.append({'image_path': img_path, 'mask_path': mask_path, 'label': cat})

df = pd.DataFrame(records)
df.to_csv('dataset_manifest.csv', index=False)
print(f"数据底表已生成，共包含 {len(df)} 个有效样本。")

# 2. 掩膜覆盖 (Mask Overlay) 极简可视化
if not df.empty:
    sample = df.iloc[0]
    img = cv2.imread(sample['image_path'], cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(sample['mask_path'], cv2.IMREAD_GRAYSCALE)
    
    plt.figure(figsize=(8, 4))
    
    # 原始超声图像
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off') 
    
    # 掩膜叠加图像
    plt.subplot(1, 2, 2)
    plt.imshow(img, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.3)
    plt.axis('off') 
    
    plt.tight_layout()
    plt.savefig('overlay_check.png', bbox_inches='tight', pad_inches=0)
    print("掩膜可视化图像 (overlay_check.png) 已生成。")
else:
    # [修改点] 增加查错提示，如果路径还是不对，可以在终端立刻看到
    print(f"警告：未找到任何有效数据！请检查路径是否真的存在: {data_dir}")