import cv2
import matplotlib.pyplot as plt

# 路径指向你发现的那个错误病例
img_path = '../data/BUSI/malignant/malignant (130).png'
mask_path = '../data/BUSI/malignant/malignant (130)_mask.png'

img = cv2.imread(img_path)
mask = cv2.imread(mask_path)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original Image (Malignant)")
plt.subplot(1, 2, 2); plt.imshow(mask); plt.title("Mask")
plt.savefig('wrong_case_130.png')
print("错误病例已可视化为 wrong_case_130.png")