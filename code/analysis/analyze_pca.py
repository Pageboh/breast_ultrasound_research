import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
df = pd.read_csv('feature_table_cleaned.csv')
X = df.iloc[:, 2:]
y = df['label']

# 2. 标准化 + PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. 绘图
plt.figure(figsize=(10, 7))
colors = {'benign': 'blue', 'malignant': 'red'}
for label, color in colors.items():
    mask = (y == label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.6, s=15)

plt.title('PCA Visualization: Benign vs Malignant')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend()
plt.savefig('pca_visualization.png')
print("PCA 降维可视化图已保存为 pca_visualization.png")