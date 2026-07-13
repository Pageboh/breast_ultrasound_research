# code1/model_factory.py
import torch
import torch.nn as nn
from torchvision import models

def get_model(model_name, num_classes=3):
    """
    模型工厂：根据名称动态加载 torchvision 模型
    :param model_name: 模型名称 (resnet18, vgg16, densenet121, mobilenet_v2, efficientnet_b0)
    :param num_classes: 输出类别数 (默认为 3，根据实际数据集调整)
    """
    
    # 统一使用最新版本的权重加载方式 (weights='DEFAULT')
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    else:
        raise ValueError(f"不支持的模型名称: {model_name}")
    
    return model

# 调试测试：确认模型工厂是否正常工作
if __name__ == '__main__':
    test_models = ['resnet18', 'vgg16', 'densenet121', 'mobilenet_v2', 'efficientnet_b0']
    for name in test_models:
        model = get_model(name)
        # 模拟一个输入张量: [batch_size, channel, height, width]
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        print(f"模型 {name} 测试通过，输出尺寸: {output.shape}")