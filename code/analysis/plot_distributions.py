import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_distributions():
    # 1. 加载清洗后的底表
    df = pd.read_csv('feature_table_cleaned.csv')
    
    # 2. 分离良恶性样本
    benign = df[df['label'] == 'benign']
    malignant = df[df['label'] == 'malignant']
    
    # 3. 选定要绘制的核心特征 (基于你之前的 Feature Importance 结果)
    features_to_plot = [
        'original_shape2D_Sphericity',
        'original_shape2D_Elongation',
        'original_firstorder_Median'
    ]
    
    # 4. 创建画板 (1行3列)
    plt.figure(figsize=(18, 5))
    
    for i, feature in enumerate(features_to_plot, 1):
        plt.subplot(1, 3, i)
        
        # 绘制直方图，设置透明度 alpha=0.6 以便观察重叠部分
        plt.hist(benign[feature], bins=30, alpha=0.6, label='Benign', color='royalblue', density=True)
        plt.hist(malignant[feature], bins=30, alpha=0.6, label='Malignant', color='crimson', density=True)
        
        # 提取特征的简写名字用于标题
        short_name = feature.split('_')[-1]
        plt.title(f'{short_name} Distribution')
        plt.xlabel('Feature Value')
        plt.ylabel('Density')
        plt.legend()
    
    # 5. 保存并提示
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300)
    print("核心特征分布对比图已成功保存为: feature_distributions.png")

if __name__ == '__main__':
    plot_feature_distributions()