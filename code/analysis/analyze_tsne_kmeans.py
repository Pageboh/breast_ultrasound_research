import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. 数据准备
df = pd.read_csv('feature_table_cleaned.csv')
X = df.iloc[:, 2:]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. t-SNE 降维 (非线性空间展示)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 3. KMeans 聚类 (观察特征空间分群)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# 4. 可视化
plt.figure(figsize=(14, 6))

# 图 A: t-SNE 原图分布
plt.subplot(1, 2, 1)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=[(1, 0, 0) if l == 'malignant' else (0, 0, 1) for l in df['label']], alpha=0.5, s=15)
plt.title('t-SNE Visualization (Colored by Label)')

# 图 B: KMeans 聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=15)
plt.title('KMeans Clustering (n=2)')

plt.tight_layout()
plt.savefig('tsne_kmeans.png')
print("t-SNE 与 KMeans 可视化图已保存为 tsne_kmeans.png")