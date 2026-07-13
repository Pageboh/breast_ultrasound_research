import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# 1. 加载清洗后的数据
df = pd.read_csv('feature_table_cleaned.csv')
X = df.iloc[:, 2:]
y = df['label'].apply(lambda x: 1 if x == 'malignant' else 0)

# 2. 重新训练 AdaBoost (使用最佳模型)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# 3. 获取特征重要性
importances = pd.Series(model.feature_importances_, index=X.columns)
top_10 = importances.nlargest(10)

# 4. 打印并绘图
print("Top 10 重要特征:")
print(top_10)

# 新增这一行：设定画板的宽度为 10，高度为 6，给长文本留出足够空间
plt.figure(figsize=(10, 6))

top_10.plot(kind='barh')
plt.title('Top 10 Important Features (AdaBoost)')
plt.tight_layout()
plt.savefig('feature_importance.png')