import os
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from train_unet import UNet 

def main():
    # 1. 路径设置
    data_dir = "/root/autodl-tmp/breast_ultrasound_research/data"
    weights_path = "/root/autodl-tmp/breast_ultrasound_research/code2/result/best_unet.pth" 
    csv_path = "/root/autodl-tmp/breast_ultrasound_research/code2/result/dice_iou_table.csv"
    output_dir = "/root/autodl-tmp/breast_ultrasound_research/code2/result/bad_cases_colored"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 从表格中自动挑出最差的 15 张图片 (Dice < 0.1 的极低分)
    df = pd.read_csv(csv_path)
    bad_df = df.sort_values(by='UNet_Dice', ascending=True).head(15)
    target_files = bad_df['Filename'].tolist()
    
    # 3. 智能寻路找原图
    true_root = data_dir
    for root, dirs, files in os.walk(data_dir):
        if 'benign' in [d.lower() for d in dirs]:
            true_root = root
            break

    # 4. 加载模型
    print("⏳ 正在加载模型重绘错误病例...")
    model = UNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    # 5. 开始画图
    with torch.no_grad():
        for filename in target_files:
            tumor_type = "malignant" if "malignant" in filename.lower() else "benign"
            img_path = os.path.join(true_root, tumor_type, filename)
            base_name, _ = os.path.splitext(filename)
            mask_path = os.path.join(true_root, tumor_type, f"{base_name}_mask.png")
            
            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                continue
                
            pil_img = Image.open(img_path).convert('RGB')
            pil_mask = Image.open(mask_path).convert('L')
            
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            mask_np = (np.array(pil_mask.resize((256, 256))) > 0).astype(np.uint8) # 真实
            pred = model(img_tensor)
            pred_np = (pred[0][0].cpu().numpy() > 0.5).astype(np.uint8) # 预测
            
            img_cv = cv2.cvtColor(np.array(pil_img.resize((256, 256))), cv2.COLOR_RGB2BGR)
            overlay = img_cv.copy()
            
            green_mask = np.zeros_like(img_cv)
            green_mask[:, :, 1] = mask_np * 255  # 绿区 GT
            red_mask = np.zeros_like(img_cv)
            red_mask[:, :, 2] = pred_np * 255    # 红区 Pred
            
            cv2.addWeighted(green_mask, 0.4, overlay, 1, 0, overlay)
            cv2.addWeighted(red_mask, 0.4, overlay, 1, 0, overlay)
            
            # 把这批图保存下来
            save_path = os.path.join(output_dir, f"Colored_BadCase_{filename}")
            cv2.imwrite(save_path, overlay)
            print(f"✅ 已生成彩色错误图: {filename}")

    print(f"\n🎉 搞定！请去 {output_dir} 查看带颜色的错误分析图！")

if __name__ == '__main__':
    main()