import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import glob

def add_overlay(image, mask, color, alpha=0.5):
    """将二值掩码以指定颜色半透明叠加到原图上"""
    if mask is None:
        return image
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    
    # 仅在存在掩码的区域进行图像混合
    overlay = image.copy()
    mask_indices = mask > 0
    overlay[mask_indices] = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)[mask_indices]
    return overlay

def main():
    parser = argparse.ArgumentParser(description="Visualize and compare segmentation masks")
    parser.add_argument('--data_dir', type=str, default='./data/BUSI', help='Path to original BUSI dataset')
    parser.add_argument('--sam_dir', type=str, default='./code3/results/sam_preds', help='Path to SAM predictions')
    parser.add_argument('--medsam_dir', type=str, default='./code3/results/medsam_preds', help='Path to MedSAM predictions')
    parser.add_argument('--save_dir', type=str, default='./code3/results/visualizations', help='Directory to save output images')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of random samples to visualize')
    parser.add_argument('--target_image', type=str, default=None, help='Specific image name to visualize (e.g., "malignant (15)")')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 收集所有的测试案例
    all_cases = []
    classes = ['benign', 'malignant']
    for cls in classes:
        cls_folder = os.path.join(args.data_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        orig_images = [f for f in glob.glob(os.path.join(cls_folder, "*.png")) if "_mask" not in f]
        for img_path in orig_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            all_cases.append((cls, base_name, img_path))
    
    if len(all_cases) == 0:
        print("No images found in data directory.")
        return

    # 筛选需要可视化的案例 (指定图片 or 随机抽取)
    if args.target_image:
        selected_cases = [c for c in all_cases if c[1] == args.target_image]
        if not selected_cases:
            print(f"Target image '{args.target_image}' not found in dataset!")
            return
    else:
        selected_cases = random.sample(all_cases, min(args.num_samples, len(all_cases)))

    print(f"Visualizing {len(selected_cases)} cases...")

    for cls, base_name, img_path in selected_cases:
        # 构建各类结果的路径
        gt_path = os.path.join(args.data_dir, cls, f"{base_name}_mask.png")
        sam_path = os.path.join(args.sam_dir, cls, f"{base_name}_pred.png")
        medsam_path = os.path.join(args.medsam_dir, cls, f"{base_name}_pred.png")

        # 读取原图并转为 RGB 格式用于 Matplotlib
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 读取掩码 (灰度图)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(gt_path) else None
        sam_mask = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(sam_path) else None
        medsam_mask = cv2.imread(medsam_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(medsam_path) else None

        # 二值化掩码处理
        gt_bin = (gt_mask > 0).astype(np.uint8) if gt_mask is not None else np.zeros_like(img_rgb[:,:,0])
        sam_bin = (sam_mask > 0).astype(np.uint8) if sam_mask is not None else np.zeros_like(img_rgb[:,:,0])
        medsam_bin = (medsam_mask > 0).astype(np.uint8) if medsam_mask is not None else np.zeros_like(img_rgb[:,:,0])

        # 在原图上生成半透明彩色叠加层
        # GT: 绿色 | SAM: 红色 | MedSAM: 蓝色
        overlay_gt = add_overlay(img_rgb, gt_bin, color=(0, 255, 0))
        overlay_sam = add_overlay(img_rgb, sam_bin, color=(255, 0, 0))
        overlay_medsam = add_overlay(img_rgb, medsam_bin, color=(0, 0, 255))

        # 绘制 1x4 的对比子图
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original: {base_name}", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(overlay_gt)
        axes[1].set_title("Ground Truth (Green)", fontsize=14)
        axes[1].axis("off")

        axes[2].imshow(overlay_sam)
        axes[2].set_title("SAM Predict (Red)", fontsize=14)
        axes[2].axis("off")

        axes[3].imshow(overlay_medsam)
        axes[3].set_title("MedSAM Predict (Blue)", fontsize=14)
        axes[3].axis("off")

        plt.tight_layout()
        save_path = os.path.join(args.save_dir, f"{cls}_{base_name}_compare.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")

    print("\nVisualization complete! Please check the output folder.")

if __name__ == '__main__':
    main()