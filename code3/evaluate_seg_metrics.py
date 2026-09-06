import os
import glob
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm

def calculate_metrics(pred_mask, gt_mask):
    """
    基于混淆矩阵计算四大核心医学图像分割指标
    返回: Dice, IoU, Sensitivity, Specificity
    """
    # 确保掩码是二值化的 (0 和 1)
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    TP = np.sum((pred == 1) & (gt == 1))
    FP = np.sum((pred == 1) & (gt == 0))
    FN = np.sum((pred == 0) & (gt == 1))
    TN = np.sum((pred == 0) & (gt == 0))
    
    # 加入 1e-8 防止除以 0
    dice = (2 * TP) / (2 * TP + FP + FN + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    sensitivity = TP / (TP + FN + 1e-8)  # 召回率/敏感度
    specificity = TN / (TN + FP + 1e-8)  # 特异度
    
    return dice, iou, sensitivity, specificity

def main():
    parser = argparse.ArgumentParser(description="Evaluate Segmentation Metrics for SAM/MedSAM")
    parser.add_argument('--gt_dir', type=str, default='./data/BUSI', help='Path to original BUSI dataset (containing GT masks)')
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to predicted masks (e.g., ./code3/results/sam_preds)')
    parser.add_argument('--output_csv', type=str, default='./code3/results/evaluation_metrics.csv', help='Path to save metrics table')
    args = parser.parse_args()

    classes = ['benign', 'malignant']
    results = []

    print(f"Evaluating predictions in: {args.pred_dir}")
    
    for cls in classes:
        pred_cls_dir = os.path.join(args.pred_dir, cls)
        gt_cls_dir = os.path.join(args.gt_dir, cls)
        
        if not os.path.exists(pred_cls_dir):
            continue
            
        pred_files = glob.glob(os.path.join(pred_cls_dir, "*_pred.png"))
        
        for pred_path in tqdm(pred_files, desc=f"Evaluating {cls}"):
            # 文件名反推规则: benign (1)_pred.png -> benign (1)_mask.png
            file_name = os.path.basename(pred_path)
            base_name = file_name.replace("_pred.png", "")
            gt_path = os.path.join(gt_cls_dir, f"{base_name}_mask.png")
            
            if not os.path.exists(gt_path):
                print(f"Warning: GT mask not found for {base_name}")
                continue
                
            # 读取灰度图
            pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            
            # 确保尺寸一致
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                
            dice, iou, sens, spec = calculate_metrics(pred_mask, gt_mask)
            
            results.append({
                'Image': base_name,
                'Class': cls,
                'Dice': dice,
                'IoU': iou,
                'Sensitivity': sens,
                'Specificity': spec
            })

    if not results:
        print("No paired images found for evaluation. Please check your directory paths.")
        return

    # 生成统计表格
    df = pd.DataFrame(results)
    
    # 计算整体平均指标
    mean_metrics = df[['Dice', 'IoU', 'Sensitivity', 'Specificity']].mean()
    
    print("\n" + "="*40)
    print("        Overall Model Performance        ")
    print("="*40)
    print(f"mDice            : {mean_metrics['Dice']:.4f}")
    print(f"mIoU             : {mean_metrics['IoU']:.4f}")
    print(f"mSensitivity     : {mean_metrics['Sensitivity']:.4f}")
    print(f"mSpecificity     : {mean_metrics['Specificity']:.4f}")
    print("="*40 + "\n")
    
    # 保存至 CSV
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Detailed image-level metrics saved to {args.output_csv}")

if __name__ == '__main__':
    main()