import torch
import torchvision.models as models

def test_model_loading():
    try:
        # 尝试加载一个预训练的 ResNet18
        print("正在尝试加载 ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        
        # 构造一张随机的假图片，模拟 224x224 RGB 图像
        # (batch_size=1, channels=3, height=224, width=224)
        dummy_input = torch.randn(1, 3, 224, 224)
        
        # 进行一次前向推理
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"模型加载成功！输出形状: {output.shape}")
        print("环境完全 OK，可以进行下一步 Dataset 编写。")
        
    except Exception as e:
        print(f"模型加载失败，请检查环境: {e}")

if __name__ == "__main__":
    test_model_loading()