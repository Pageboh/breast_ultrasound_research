import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. 加载数据
df = pd.read_csv('feature_table_cleaned.csv')
X = df.iloc[:, 2:]
y_true = df['label'].apply(lambda x: 1 if x == 'malignant' else 0)

# 2. 训练并预测
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = AdaBoostClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y_true)
df['prediction'] = model.predict(X_scaled)

# 3. 筛选错误病例
wrong_cases = df[df['prediction'] != y_true]
wrong_cases[['image_path', 'label', 'prediction']].to_csv('wrong_cases.csv', index=False)

print(f"共发现 {len(wrong_cases)} 个错误分类样本，详情见 wrong_cases.csv")