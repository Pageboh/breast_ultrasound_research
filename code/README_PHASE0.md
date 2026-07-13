# Phase 0: Data Preparation & Radiomics Feature Engineering

## 1. 数据概览
*   **数据集来源**: BUSI (Breast Ultrasound Images)
*   **原始样本总量**: 646 (包含良性、恶性、正常)
*   **处理策略**: 剔除 'normal' 文件夹，仅保留 'benign' 与 'malignant' 进行二分类任务。

## 2. 处理流程 (Pipeline)
1.  **Mask 映射**: 验证每个 `.png` 图像是否有对应的 `_mask.png`。
2.  **特征提取**: 使用 `PyRadiomics` 提取 `shape2D`, `firstorder`, `glcm`, `glrlm` 等类别特征。
3.  **异常值清洗**:
    *   `inf` 值替换为 `NaN`。
    *   采用 `fillna(0)` 填充缺失特征值。
4.  **标准化**: 使用 `StandardScaler` 对特征矩阵进行 Z-score 标准化。

## 3. 重要文件映射
*   `extract_features.py`: 执行特征批量提取。
*   `clean_data.py`: 执行异常值剔除与标准化。
*   `feature_table_cleaned.csv`: 最终生成的清洗后特征矩阵。

## 4. 关键记录
*   **日期**: 2026-04-XX
*   **特征维度**: 102 维
*   **备注**: 剔除异常样本 1 例，最终有效样本 645 个。