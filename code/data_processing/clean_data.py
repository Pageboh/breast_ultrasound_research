import pandas as pd
import numpy as np

def clean_radiomics_data():
    # 1. 读取原始特征表
    input_file = 'radiomics_features.csv'
    output_file = 'feature_table_cleaned.csv'
    
    print(f"正在读取特征表: {input_file}...")
    df = pd.read_csv(input_file)
    
    # 2. 识别数值列（跳过 image_path 和 label）
    # 假设前两列分别是 image_path 和 label
    features = df.iloc[:, 2:]
    
    # 3. 处理异常值 (NaN 和 inf)
    # 将 inf 替换为 NaN，然后将所有 NaN 填充为 0
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(0)
    
    # 4. 合并标签列，保存清洗后的数据
    cleaned_df = pd.concat([df[['image_path', 'label']], features], axis=1)
    cleaned_df.to_csv(output_file, index=False)
    
    print("-" * 40)
    print("数据清洗完成")
    print(f"清洗后样本数: {len(cleaned_df)}")
    print(f"清洗后维度:   {cleaned_df.shape[1]} 列 (含路径和标签)")
    print(f"输出文件:     {output_file}")
    print("-" * 40)

if __name__ == '__main__':
    clean_radiomics_data()