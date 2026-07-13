import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings('ignore')

def generate_markdown_summary(df, feature_cols, output_path='output/feature_summary.md'):
    """生成特征的描述性统计摘要并保存为 Markdown 表格"""
    summary = df[feature_cols].describe().T
    summary = summary[['mean', 'std', 'min', 'max']]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 影像组学特征统计摘要 (Radiomics Features Summary)\n\n")
        f.write("| Feature Name | Mean | Std | Min | Max |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for index, row in summary.iterrows():
            f.write(f"| {index} | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")
    print(f"✅ 成功生成统计摘要: {output_path}")

def run_feature_selection():
    # 1. 数据加载
    # 1. 数据加载 (去 output 文件夹里找)
    df = pd.read_csv('output/feature_table_cleaned.csv')
    feature_cols = df.columns[2:]  # 假设前两列是 image_path 和 label
    X = df[feature_cols]
    y = df['label'].apply(lambda x: 1 if x == 'malignant' else 0)
    
    print(f"初始特征维度: {X.shape[1]}")

    # ================= 产出物 1: feature_summary.md =================
    generate_markdown_summary(df, feature_cols)

    # ================= 步骤 1: 方差过滤 (Variance Thresholding) =================
    # 剔除在所有样本中数值几乎不变的特征 (阈值设为 1e-5)
    vt = VarianceThreshold(threshold=1e-5)
    vt.fit(X)
    features_after_var = X.columns[vt.get_support()]
    X_var = X[features_after_var]
    print(f"方差过滤后剩余特征数: {X_var.shape[1]}")

    # ================= 步骤 2: 相关性过滤 (Pearson Correlation) =================
    # 剔除高度冗余的特征对 (阈值设为 |r| > 0.9)
    corr_matrix = X_var.corr().abs()
    # 提取上三角矩阵
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # 找到相关系数大于0.9的特征
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    features_after_corr = [col for col in X_var.columns if col not in to_drop]
    X_corr = X_var[features_after_corr]
    print(f"相关性过滤后剩余特征数: {X_corr.shape[1]}")

    # ================= 步骤 3: LASSO 筛选 (L1 Regularization) =================
    # LASSO 对数值尺度敏感，必须先进行标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_corr)

    # 使用带 L1 惩罚项的 Logistic 回归 (LASSO 逻辑回归)
    # CV=5 表示 5 折交叉验证，自动寻找最佳正则化强度
    lasso = LogisticRegressionCV(cv=5, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
    lasso.fit(X_scaled, y)

    # 提取非零权重的特征
    selected_mask = lasso.coef_[0] != 0
    final_features = np.array(features_after_corr)[selected_mask]
    print(f"LASSO 筛选后最终保留核心特征数: {len(final_features)}")

   # ================= 产出物 2: selected_features.csv =================
    # 将最终保留的特征与基础信息合并
    final_df = pd.concat([df[['image_path', 'label']], df[final_features]], axis=1)
    final_df.to_csv('output/selected_features.csv', index=False)
    print("✅ 成功生成精简特征底表: output/selected_features.csv")
    
    print("\n🎉 阶段 1 学习目标全部达成！")

if __name__ == '__main__':
    run_feature_selection()