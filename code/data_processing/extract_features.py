import os
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import logging

# 屏蔽底层库的冗余日志，保持终端呈现极简和专业
logging.getLogger('radiomics').setLevel(logging.ERROR)

def build_radiomics_matrix():
    print("开始构建影像组学特征阵列，进入研究与分析流程...")
    
    # === 修正后的读取逻辑：确保路径唯一且安全 ===
    manifest_path = 'output/dataset_manifest.csv'
    if not os.path.exists(manifest_path):
        print(f"错误：未找到基础底表 {manifest_path}")
        return
        
    df = pd.read_csv(manifest_path)
    # =======================================
    
    # 针对超声图像的 2D 属性进行专项配置
    settings = {
        'binWidth': 25,
        'resampledPixelSpacing': None, 
        'interpolator': sitk.sitkBSpline,
        'force2D': True,
        'force2Ddimension': 0,
        'label': 255  # 明确指定病灶区域的像素值为 255
    }
    
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    
    # 精准激活大纲要求的二维形状与所有高维纹理特征族
    feature_classes = ['shape2D', 'firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm']
    for fc in feature_classes:
        extractor.enableFeatureClassByName(fc)
        
    results = []
    total_samples = len(df)
    
    for index, row in df.iterrows():
        img_path = row['image_path']
        mask_path = row['mask_path']
        
        try:
            # 1. 图像与掩膜载入
            img_sitk = sitk.ReadImage(img_path)
            mask_sitk = sitk.ReadImage(mask_path)
            
            # 2. 预处理：降维至单通道 (Grayscale Conversion)
            if img_sitk.GetNumberOfComponentsPerPixel() > 1:
                img_sitk = sitk.VectorIndexSelectionCast(img_sitk, 0)
            if mask_sitk.GetNumberOfComponentsPerPixel() > 1:
                mask_sitk = sitk.VectorIndexSelectionCast(mask_sitk, 0)

            # 3. 影像组学特征提取 (Radiomics Extraction)
            feature_vector = extractor.execute(img_sitk, mask_sitk)
            
            clean_features = {
                'image_path': img_path,
                'label': row['label']
            }
            
            for key, value in feature_vector.items():
                if not key.startswith('diagnostics_'):
                    clean_features[key] = float(value)
            
            results.append(clean_features)
            
            if (index + 1) % 50 == 0 or (index + 1) == total_samples:
                print(f"计算进度: {index + 1}/{total_samples}")
                
        except Exception as e:
            # 4. 鲁棒性保护：异常拦截 (Exception Trapping & Pipeline Logging)
            # 目的：确保单个微小或失效掩膜不会导致整个批处理流水线中断
            print(f"Pipeline Warning: Skipping invalid sample [{os.path.basename(img_path)}]. Reason: {e}")
            continue # 主动剔除无效数据，维持流水线运行
            
    # 保存结果
    results_df = pd.DataFrame(results)
    output_csv = 'output/radiomics_features.csv'
    results_df.to_csv(output_csv, index=False)
    
    num_features = results_df.shape[1] - 2 if not results_df.empty else 0
    print("-" * 40)
    print("研究特征阵列构建完成")
    print(f"有效样本数: {len(results_df)}")
    print(f"特征维度:   {num_features} 个维度的影像组学特征")
    print(f"输出路径:   {output_csv}")
    print("-" * 40)

if __name__ == '__main__':
    build_radiomics_matrix()