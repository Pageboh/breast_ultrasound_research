import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc

def run_ml_benchmark():
    # 1. 强制定位到当前目录下的 output 文件夹
    # 只要你是在 code/ 目录下运行此脚本，下面的路径就永远有效
    input_path = 'output/feature_table_cleaned.csv'
    
    if not os.path.exists(input_path):
        print(f"❌ 错误：在当前目录下找不到文件 {input_path}")
        print(f"当前工作目录是: {os.getcwd()}")
        return

    df = pd.read_csv(input_path)
    # ... 后续代码不变
    # ... 后续读取和保存路径也都加上 os.path.join(base_dir, 'output', '...')
    X = df.iloc[:, 2:]
    y = df['label'].apply(lambda x: 1 if x == 'malignant' else 0)

    # 2. 划分集与标准化
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 3. 定义模型
    models = {
        "LogisticRegression": LogisticRegression(),
        "SVM": SVC(probability=True),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier()
    }

    # 4. 训练与绘图
    plt.figure(figsize=(10, 8))
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        results.append({'Model': name, 'Accuracy': acc, 'AUC': auc_score})

        # 绘制 ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        if name == 'AdaBoost':
            plt.plot(fpr, tpr, color='red', linewidth=3, zorder=10, label=f'{name} (AUC={auc_score:.3f})')
        else:
            plt.plot(fpr, tpr, color='gray', alpha=0.4, linewidth=1.5, label=f'{name} (AUC={auc_score:.3f})')

    # 5. 完成绘图与保存
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title('ROC Curve Comparison: Multi-Model Benchmark')
    plt.legend(loc="lower right")
    plt.savefig('output/roc_all_models.png', dpi=300, bbox_inches='tight')
    pd.DataFrame(results).to_csv('output/model_comparison.csv', index=False)
    print("✅ 训练与评估完成，结果已存入 output/")

if __name__ == '__main__':
    run_ml_benchmark()