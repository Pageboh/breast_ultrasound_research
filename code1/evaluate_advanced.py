import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import BUSIDataset
from model_factory import get_model
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. 函数定义
def get_target_layer(model, name):
    if 'resnet' in name: return model.layer4[-1]
    if 'vgg' in name: return model.features[-1]
    if 'densenet' in name: return model.features.norm5
    if 'mobilenet' in name: return model.features[-1]
    if 'efficientnet' in name: return model.features[-1]
    return None

# 2. 配置环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '/root/autodl-tmp/breast_ultrasound_research/data/BUSI'
checkpoint_dir = '/root/autodl-tmp/breast_ultrasound_research/checkpoints'
gradcam_dir = '/root/autodl-tmp/breast_ultrasound_research/gradcam_examples'
os.makedirs(gradcam_dir, exist_ok=True)
models_to_eval = ['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2', 'efficientnet_b0']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_loader = DataLoader(BUSIDataset(data_path, transform=transform), batch_size=32, shuffle=False)

results_for_plot = {}

# 3. 执行评估
for name in models_to_eval:
    print(f"\n================ 正在评估: {name} ================")
    model = get_model(name, num_classes=2).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f'{name}_best.pth'), map_location=device))
    model.eval()
    
    # 3.1 指标收集 (无梯度)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            probs = torch.softmax(model(inputs), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    results_for_plot[name] = {'probs': all_probs, 'labels': all_labels}
    
    # 3.2 Grad-CAM 可视化 (需梯度)
    cam = GradCAM(model=model, target_layers=[get_target_layer(model, name)])
    
    # 获取第一张图
    input_sample = next(iter(val_loader))[0][0:1].to(device)
    input_sample.requires_grad = True
    
    grayscale_cam = cam(input_tensor=input_sample, targets=None)
    img_np = input_sample[0].detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # === 保存对应的原图并打印真实标签（只存一次即可） ===
    orig_img_path = os.path.join(gradcam_dir, 'original_image.png')
    if not os.path.exists(orig_img_path):
        cv2.imwrite(orig_img_path, cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        true_label = next(iter(val_loader))[1][0].item()
        label_name = 'Malignant (恶性)' if true_label == 1 else 'Benign (良性)'
        print(f"🌟 注意：当前 Grad-CAM 使用的原图已保存为 'original_image.png'")
        print(f"🌟 这张图的真实标签是: {label_name}")
    # ==========================================================

    vis = show_cam_on_image(img_np, grayscale_cam[0, :], use_rgb=True)
    cv2.imwrite(os.path.join(gradcam_dir, f'{name}_cam.png'), cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"✅ {name} 的 Grad-CAM 热力图已保存。")

# 4. ROC/PR 绘图
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, res in results_for_plot.items():
    fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--'); plt.title('ROC Curve'); plt.legend()

plt.subplot(1, 2, 2)
for name, res in results_for_plot.items():
    precision, recall, _ = precision_recall_curve(res['labels'], res['probs'])
    plt.plot(recall, precision, label=f'{name}')
plt.title('PR Curve'); plt.legend()

plt.tight_layout()
plt.savefig('roc_pr_all_cnn.png')
print("\n✅ ROC 和 PR 曲线已保存至 roc_pr_all_cnn.png")

# 5. 混淆矩阵绘图 (1x5 布局，并排展示五个模型)
fig, axes = plt.subplots(1, len(models_to_eval), figsize=(20, 4))
for i, name in enumerate(models_to_eval):
    res = results_for_plot[name]
    preds = (np.array(res['probs']) >= 0.5).astype(int) 
    cm = confusion_matrix(res['labels'], preds)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f'{name}')
    
plt.tight_layout()
plt.savefig('confusion_matrix_all_cnn.png')
print("✅ 5个模型的混淆矩阵已保存至 confusion_matrix_all_cnn.png")

print("\n=== 任务完成：所有图表及热力图已生成 ===")