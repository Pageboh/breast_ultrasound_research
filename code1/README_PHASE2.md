# Breast Ultrasound Research (乳腺超声辅助诊断研究)

## 📌 项目概述
本研究旨在通过深度学习方法对乳腺超声图像进行良恶性分类与特征分析。项目涵盖 CNN 模型分类、MedSAM 分割及机器学习分析。

## 📁 目录结构说明
- `data/`: 原始 BUSI 数据集
- `checkpoints/`: 已训练的模型权重 (.pth)
- `results/`: 实验分析产出
  - `gradcam/`: 模型可解释性热力图
  - `plots/`: ROC/PR 曲线及 Loss 曲线
- `code/`: 机器学习实现
- `code1/`: CNN 深度学习核心模块
- `code2/`: MedSAM 模型分割逻辑

## 📊 实验成果 (Phase 2)
本阶段已完成 CNN 系列模型的性能评估，包含：
1. **性能对比**: 生成了各模型的 ROC 与 PR 曲线。
2. **可视化验证**: 利用 Grad-CAM 展示模型对病灶区域的关注度。

## 🚀 如何运行
若需更新 CNN 模型评估结果，请进入 `code1/` 目录运行：
```bash
python evaluate_advanced.py
---

