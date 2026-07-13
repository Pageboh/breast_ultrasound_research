# Phase 1: Machine Learning Classifier Benchmark

## 1. 实验目的
本阶段旨在验证影像组学特征在乳腺超声良恶性鉴别任务中的有效性，通过对比多种机器学习分类器，筛选最优模型并进行误差分析。

## 2. 实验记录日志
| Model | Parameters | Accuracy | AUC | Note |
| :--- | :--- | :--- | :--- | :--- |
| LogisticRegression | C=1.0, solver='liblinear' | 0.9457 | 0.9877 | Baseline |
| SVM | C=1.0, kernel='rbf' | 0.9302 | 0.9847 | Standard config |
| DecisionTree | max_depth=None | 0.9147 | 0.9241 | Overfitting risk |
| RandomForest | n_estimators=100 | 0.9147 | 0.9775 | - |
| KNN | n_neighbors=5 | 0.8527 | 0.8800 | - |
| NaiveBayes | Gaussian | 0.7907 | 0.8528 | - |
| AdaBoost | n_estimators=100 | **0.9767** | **0.9924** | **Best Performance** |

## 3. 关键结论
*   **最佳模型**：AdaBoost 表现最优，AUC 高达 0.9924。
*   **核心特征**：`original_shape2D_Sphericity` 与 `original_shape2D_Elongation` 为最显著的判别依据。
*   **误差分析**：主要存在 1 例 False Negative（`malignant (130).png`），经可视化分析，该病例形态特征与良性结节高度相似，属于组学重叠区域。

## 模型训练与评估流水线
- **脚本**: `train_and_evaluate.py`
- **功能**: 执行全量 7 种基准模型的训练、评估指标计算以及对比 ROC 曲线生成。
- **产出物**:
    - `output/model_comparison.csv`: 各模型准确率与 AUC 指标。
    - `output/roc_all_models.png`: 包含 AdaBoost 高光突出的多模型 ROC 对比图。
    
## 4. 产出物清单 (Artifacts)
*   `feature_table_cleaned.csv`: 标准化清洗后的数据集。
*   `feature_importance.png`: 影响因子可视化。
*   `pca_visualization.png`: 低维空间类间分布。
*   `roc_all_models.png`: 模型性能对比曲线。
*   `wrong_cases.csv`: 误差样本记录。