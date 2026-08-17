import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 路径设置
    data_dir = "/root/autodl-tmp/breast_ultrasound_research/data"
    result_dir = "/root/autodl-tmp/breast_ultrasound_research/code2/result"
    csv_path = os.path.join(result_dir, "dice_iou_table.csv")
    bad_cases_dir = os.path.join(result_dir, "bad_segmentation_cases")
    
    os.makedirs(bad_cases_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print("❌ 找不到 dice_iou_table.csv，请确认上一阶段训练脚本已成功执行！")
        return

    # 1. 读取数据并解析良恶性标签
    df = pd.read_csv(csv_path)
    # 根据文件名包含的字符串判断是 benign 还是 malignant
    df['Tumor_Type'] = df['Filename'].apply(lambda x: 'Benign' if 'benign' in x.lower() else 'Malignant')

    # ==========================================
    # 产出 1：绘制 Dice 和 IoU 的箱线图 (Boxplot)
    # ==========================================
    # 整理数据为 seaborn 需要的长格式 (Melt)
    dice_cols = [col for col in df.columns if 'Dice' in col]
    df_melt = pd.melt(df, id_vars=['Filename', 'Tumor_Type'], value_vars=dice_cols, 
                      var_name='Model', value_name='Dice_Score')
    df_melt['Model'] = df_melt['Model'].str.replace('_Dice', '') # 清理列名，只留模型名

    # 设置绘图风格并画图
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_melt, x='Model', y='Dice_Score', hue='Tumor_Type', palette="Set2")
    plt.title('Segmentation Performance: Benign vs Malignant across Models')
    plt.ylabel('Dice Coefficient')
    plt.ylim(0, 1.1)
    
    # 保存图片
    boxplot_path = os.path.join(result_dir, "dice_iou_boxplot.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 箱线图已保存至: {boxplot_path}")

    # ==========================================
    # 产出 2：Benign 与 Malignant 分割难度统计对比
    # ==========================================
    print("\n📊 【Benign vs Malignant 分割难度对比】")
    grouped = df_melt.groupby(['Model', 'Tumor_Type'])['Dice_Score'].mean().unstack()
    print(grouped.to_string())
    print("结论辅助：通常 Malignant (恶性) 的平均 Dice 会低于 Benign (良性)，因为恶性肿瘤形态不规则、边界浸润更严重。")

    # ==========================================
    # 产出 3：提取 Bad Cases (分割错误分析素材)
    # ==========================================
    # 我们以最基础的 U-Net 为基准，找出 Dice 得分最低的前 15 个病例
    bad_cases_df = df.sort_values(by='UNet_Dice').head(15)
    bad_cases_csv_path = os.path.join(bad_cases_dir, "bad_cases_summary.csv")
    bad_cases_df.to_csv(bad_cases_csv_path, index=False)
    
    print(f"\n🔍 正在提取 {len(bad_cases_df)} 个典型的 Bad Cases 用于人工分析...")
    
    # 智能寻路寻找原始图片(复用之前逻辑)
    true_root = data_dir
    for root, dirs, files in os.walk(data_dir):
        dirs_lower = [d.lower() for d in dirs]
        if 'benign' in dirs_lower and 'malignant' in dirs_lower:
            true_root = root
            break

    # 把这 15 个差等生的原图和 Mask 复制到专用文件夹中，方便你阅片
    for _, row in bad_cases_df.iterrows():
        filename = row['Filename']
        tumor_type = row['Tumor_Type'].lower()
        src_img_path = os.path.join(true_root, tumor_type, filename)
        
        base_name, ext = os.path.splitext(filename)
        src_mask_path = os.path.join(true_root, tumor_type, f"{base_name}_mask.png")
        
        if os.path.exists(src_img_path) and os.path.exists(src_mask_path):
            # 拷贝过去并重命名加上得分，方便直观查看
            score = row['UNet_Dice']
            shutil.copy(src_img_path, os.path.join(bad_cases_dir, f"DICE_{score:.2f}_{filename}"))
            shutil.copy(src_mask_path, os.path.join(bad_cases_dir, f"GT_MASK_{score:.2f}_{filename}"))

    print(f"✅ Bad Cases 素材已全部放入: {bad_cases_dir}/ 文件夹，请打开图片结合医学特征进行归因！")

if __name__ == "__main__":
    main()