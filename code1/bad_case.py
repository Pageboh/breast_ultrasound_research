import os
import torch
import numpy as np
import cv2
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import BUSIDataset
from model_factory import get_model

# === 新增导入的库 ===
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# === 新增获取 Target Layer 的函数 ===
def get_target_layer(model, name):
    if 'resnet' in name: return model.layer4[-1]
    if 'vgg' in name: return model.features[-1]
    if 'densenet' in name: return model.features.norm5
    if 'mobilenet' in name: return model.features[-1]
    if 'efficientnet' in name: return model.features[-1]
    return None

def extract_bad_cases():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = '/root/autodl-tmp/breast_ultrasound_research/data/BUSI'
    checkpoint_dir = '/root/autodl-tmp/breast_ultrasound_research/checkpoints'
    bad_cases_dir = '/root/autodl-tmp/breast_ultrasound_research/bad_cases'
    os.makedirs(bad_cases_dir, exist_ok=True)
    
    # 选取你表现最好的模型来进行分析，这里以 densenet121 为例
    best_model_name = 'densenet121'
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 假设你的 dataset 能返回原图路径或名称最好，如果没有，我们就直接保存 Tensor 还原的图
    val_dataset = BUSIDataset(data_path, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    print(f"正在加载 {best_model_name} 进行错误病例诊断...")
    model = get_model(best_model_name, num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{best_model_name}_best.pth'), map_location=device))
    model.eval()
    
    # === 新增：初始化 GradCAM ===
    target_layer = get_target_layer(model, best_model_name)
    cam = GradCAM(model=model, target_layers=[target_layer])

    fp_count = 0  # 假阳性：良性(0)被误判为恶性(1)
    fn_count = 0  # 假阴性：恶性(1)被误判为良性(0)
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 获取预测概率和最终预测类别 (阈值设为 0.5)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs > 0.5).int()
            
            label_val = labels.item()
            pred_val = preds.item()
            
            if pred_val != label_val:
                # 还原归一化操作，转为可以显示的 BGR 图像格式
                img_tensor = inputs[0].cpu().clone()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_unnorm = img_tensor * std + mean
                
                img_np = img_unnorm.permute(1, 2, 0).numpy()
                img_np_clip = np.clip(img_np * 255, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np_clip, cv2.COLOR_RGB2BGR)
                
                # 分类保存：是假阳性还是假阴性
                if label_val == 0 and pred_val == 1:
                    filename = os.path.join(bad_cases_dir, f'FP_idx{idx}_prob{probs.item():.2f}.png')
                    cam_filename = os.path.join(bad_cases_dir, f'FP_idx{idx}_prob{probs.item():.2f}_cam.png') # 新增的热力图文件名
                    fp_count += 1
                else:
                    filename = os.path.join(bad_cases_dir, f'FN_idx{idx}_prob{probs.item():.2f}.png')
                    cam_filename = os.path.join(bad_cases_dir, f'FN_idx{idx}_prob{probs.item():.2f}_cam.png') # 新增的热力图文件名
                    fn_count += 1
                
                # 1. 保存原本的错误病例图像 (未修改原功能)
                cv2.imwrite(filename, img_bgr)
                
                # === 2. 新增：计算并保存当前错误病例的 Grad-CAM 热力图 ===
                # 临时开启梯度以满足 Grad-CAM 计算条件
                with torch.enable_grad():
                    input_tensor = inputs.clone().requires_grad_(True)
                    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                    
                    # 将还原后的 numpy 图像归一化至 0~1 用于叠加热力图
                    img_for_cam = (img_np_clip / 255.0).astype(np.float32)
                    vis = show_cam_on_image(img_for_cam, grayscale_cam[0, :], use_rgb=True)
                    
                    # 保存热力图
                    cv2.imwrite(cam_filename, cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                # ========================================================

    print("=== 错误病例诊断完成 ===")
    print(f"总计提取到: {fp_count} 个假阳性 (良性误判为恶性)")
    print(f"总计提取到: {fn_count} 个假阴性 (恶性误判为良性)")
    print(f"原图及对应的 Grad-CAM 热力图已保存至: {bad_cases_dir}")

if __name__ == "__main__":
    extract_bad_cases()