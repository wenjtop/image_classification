
"""
原始图片路径：
├─ images
│  ├── cat
│  └── dog
"""

import os
import yaml
import shutil
import random

# 定义原始图像文件夹路径
ratio = 0.9
original_images_folder = 'catdog/train'
new_images_folder = 'new_images'

os.makedirs(new_images_folder, exist_ok=True)
# 定义目标文件夹路径
train_folder = os.path.join(new_images_folder, 'train')
val_folder = os.path.join(new_images_folder, 'val')

# 创建 train 和 val 文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

category_dict = {}
# 创建 train 和 val 文件夹下的子文件夹
subfolders = os.listdir(original_images_folder)
for index, subfolder in enumerate(subfolders):
    os.makedirs(os.path.join(train_folder, subfolder), exist_ok=True)
    os.makedirs(os.path.join(val_folder, subfolder), exist_ok=True)
    category_dict[subfolder] = index

with open(os.path.join(new_images_folder, 'class_name.yaml'), 'w') as file:
    yaml.dump(category_dict, file)

# 遍历原始文件夹中的子文件夹
for subfolder in subfolders:
    original_subfolder_path = os.path.join(original_images_folder, subfolder)
    files = os.listdir(original_subfolder_path)

    # 打乱文件列表
    random.shuffle(files)

    # 计算分割点
    split_point = int(len(files) * ratio)

    # 分割文件
    train_files = files[:split_point]
    val_files = files[split_point:]

    # 移动文件到 train 和 val 文件夹
    for filename in train_files:
        src = os.path.join(original_subfolder_path, filename)
        dst = os.path.join(train_folder, subfolder, filename)
        shutil.copy(src, dst)

    for filename in val_files:
        src = os.path.join(original_subfolder_path, filename)
        dst = os.path.join(val_folder, subfolder, filename)
        shutil.copy(src, dst)

print("图片已成功划分并移动到 train 和 val 文件夹。")

