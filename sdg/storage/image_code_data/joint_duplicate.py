import os
import pandas as pd

def evaluate_joint_duplicate(duplicate_code_files, duplicate_image_files, csv_file):
    """联合查重：只有图像和代码都被认为是重复的，才认为它们是重复的"""
    # 读取CSV文件，获取图像和代码的配对关系
    df = pd.read_csv(csv_file)
    # print(duplicate_image_files)
    # print(duplicate_code_files)
    total_pairs = len(df)
    total_duplicates = 0
    duplicate_codes = set()  # 使用集合自动去重
    # 遍历每对配对，检查图像和代码是否都在重复列表中
    for _, row in df.iterrows():
        image_filename = row['image_file']
        code_filename = row['code_file']
        # print(image_filename)
        # print(code_filename)
        # 判断图像和代码是否都为重复
        if image_filename in duplicate_image_files and code_filename in duplicate_code_files:
            total_duplicates += 1
            duplicate_codes.add(code_filename)

    # 计算不重复率
    non_duplicate_rate = (1 - total_duplicates / total_pairs) * 100
    return non_duplicate_rate,list(duplicate_codes)


