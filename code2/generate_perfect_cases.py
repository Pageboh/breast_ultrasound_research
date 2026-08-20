import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# 从你实际跑出权重的脚本中导入正确尺寸的 UNet 模型
from train_unet import UNet 

def main():
    # 1. 路径与基础设置
    data_dir = "/root/autodl-tmp/breast_ultrasound_research/data"
    weights_path = "/root/autodl-tmp/breast_ultrasound_research/code2/result/best_unet.pth" 
    output_dir = "/root/autodl-tmp/breast_ultrasound_research/code2/result/perfect_cases"
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 智能寻路：自动寻找真正包含 benign 和 malignant 图片的子文件夹路径
    true_root = data_dir
    for root, dirs, files in os.walk(data_dir):
        dirs_lower = [d.lower() for d in dirs]
        if 'benign' in dirs_lower and 'malignant' in dirs_lower:
            true_root = root
            break

    # 你筛选出的高分王者列表
    target_files = [
        "malignant (120).png", "benign (140).png", "benign (111).png", 
        "benign (284).png", "benign (403).png", "benign (73).png", 
        "benign (324).png", "benign (289).png", "benign (125).png", 
        "benign (17).png", "benign (50).png", "benign (241).png", 
        "benign (231).png", "benign (67).png", "benign (191).png"
    ]
    
    # 2. 加载训练好的模型
    print("⏳ 正在加载 U-Net 模型...")
    model = UNet().to(device)
    if not os.path.exists(weights_path):
        print(f"❌ 找不到权重文件: {weights_path} (请检查你的权重保存名称)")
        return
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    # 3. 图像预处理规则 (必须和训练时完全一致)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    
    print(f"🔍 开始生成 {len(target_files)} 张高分样本的可视化图...")
    
    # 4. 开始推理与画图
    with torch.no_grad():
        for filename in target_files:
            tumor_type = "malignant" if "malignant" in filename.lower() else "benign"
            
            # 使用自动寻路得到的 true_root 拼接原图和真实 Mask 的路径
            img_path = os.path.join(true_root, tumor_type, filename)
            base_name, _ = os.path.splitext(filename)
            mask_path = os.path.join(true_root, tumor_type, f"{base_name}_mask.png")
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"⚠️ 跳过: 找不到图片 {filename}")
                continue
                
            # 读取图片并转为 Tensor
            pil_img = Image.open(img_path).convert('RGB')
            pil_mask = Image.open(mask_path).convert('L')
            
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            mask_tensor = transform(pil_mask)
            mask_np = (mask_tensor[0].cpu().numpy() > 0).astype(np.uint8) # 真实 Mask
            
            # 模型预测 (概率 > 0.5 视为病灶)
            pred = model(img_tensor)
            pred_np = (pred[0][0].cpu().numpy() > 0.5).astype(np.uint8) # 预测 Mask
            
            # === 开始绘制炫酷的 Overlay ===
            # 将原图转为 BGR 格式（OpenCV 专用）
            img_cv = cv2.cvtColor(np.array(pil_img.resize((256, 256))), cv2.COLOR_RGB2BGR)
            overlay = img_cv.copy()
            
            # 绿色代表 Ground Truth (医生画的)
            green_mask = np.zeros_like(img_cv)
            green_mask[:, :, 1] = mask_np * 255  
            
            # 红色代表 Prediction (AI 画的)
            red_mask = np.zeros_like(img_cv)
            red_mask[:, :, 2] = pred_np * 255    
            
            # 将绿色和红色半透明叠加上去 (重合的地方会自动变成黄色)
            cv2.addWeighted(green_mask, 0.4, overlay, 1, 0, overlay)
            cv2.addWeighted(red_mask, 0.4, overlay, 1, 0, overlay)
            
            save_path = os.path.join(output_dir, f"HighScore_{filename}")
            cv2.imwrite(save_path, overlay)
            print(f"✅ 已生成: {filename}")

    print(f"\n🎉 全部完成！快去 {output_dir} 文件夹下挑选放入 PPT 吧！")

if __name__ == '__main__':
    main()