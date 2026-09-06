import os
import glob
import argparse
import numpy as np
from PIL import Image
import cv2
import torch
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

def get_busi_pairs(data_dir):
    """精准匹配 BUSI 数据集的原图与 Mask 路径"""
    pairs = []
    classes = ['benign', 'malignant']
    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        if not os.path.exists(cls_folder):
            continue
        all_files = glob.glob(os.path.join(cls_folder, "*.png"))
        orig_images = [f for f in all_files if "_mask" not in os.path.basename(f)]
        for img_path in orig_images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(cls_folder, f"{base_name}_mask.png")
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path, cls, base_name))
    return pairs

def extract_bounding_box(mask_path, padding=5):
    """根据 GT Mask 自动计算并扩展外接矩形作为 Box Prompt"""
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    
    y_indices, x_indices = np.where(mask_np > 0)
    if len(y_indices) == 0 or len(x_indices) == 0:
        return None
        
    xmin = np.max([0, np.min(x_indices) - padding])
    ymin = np.max([0, np.min(y_indices) - padding])
    xmax = np.min([mask_np.shape[1], np.max(x_indices) + padding])
    ymax = np.min([mask_np.shape[0], np.max(y_indices) + padding])
    
    return np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="SAM / MedSAM Zero-shot Inference on BUSI")
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to BUSI dataset (修改为你实际的 data 路径)')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model weight (.pth)')
    parser.add_argument('--model_type', type=str, default='vit_b', help='SAM model type')
    parser.add_argument('--output_dir', type=str, default='./code3/results/preds', help='Path to save predicted masks')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 加载模型
    print(f"Loading model weights from {args.checkpoint}...")
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # 2. 获取数据对
    pairs = get_busi_pairs(args.data_dir)
    print(f"Found {len(pairs)} image-mask pairs for inference.")

    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 循环推理
    for img_path, mask_path, cls_name, base_name in tqdm(pairs, desc="Running Inference"):
        image = cv2.imread(img_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 提取 Box Prompt
        box = extract_bounding_box(mask_path, padding=5)
        if box is None:
            continue

        # 模型推理
        predictor.set_image(image_rgb)
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=False,
        )

        pred_mask = (masks[0].astype(np.uint8)) * 255

        # 保存预测结果
        save_cls_dir = os.path.join(args.output_dir, cls_name)
        os.makedirs(save_cls_dir, exist_ok=True)
        save_path = os.path.join(save_cls_dir, f"{base_name}_pred.png")
        cv2.imwrite(save_path, pred_mask)

    print(f"Inference completed! Predictions successfully saved to {args.output_dir}")

if __name__ == '__main__':
    main()