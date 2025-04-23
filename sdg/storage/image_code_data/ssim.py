import os

import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
from PIL import Image

def calculate_ssim(image1_path, image2_path):
    # image1_path = image1_path.encode('utf-8').decode('gbk')
    # image2_path = image2_path.encode('utf-8').decode('gbk')
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)


    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    ssim_value, _ = ssim(image1, image2, full=True)
    # print(ssim_value)
    return (int)(ssim_value*100)


def build_code_mapping(csv_path):
    """构建代码文件到原始图像的映射字典"""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['code', 'image'])
    return dict(zip(df['code'], df['image']))


def process_screenshots(screenshot_folder, image_folder, code_map):
    """处理所有渲染截图并计算SSIM"""
    score_dict = {}
    # print(screenshot_folder)
    # print(image_folder)
    for screenshot_name in os.listdir(screenshot_folder):
        # 获取对应的代码文件名
        code_file = os.path.splitext(screenshot_name)[0] + '.json'

        # 查找原始图像文件名
        original_image = code_map.get(code_file)
        # print(original_image)
        if not original_image:
            continue

        # 构建完整路径
        screenshot_path = os.path.join(screenshot_folder, screenshot_name)
        original_path = os.path.join(image_folder, original_image)
        # print(screenshot_path)
        # print(original_path)
        # 计算SSIM
        if os.path.exists(original_path):
            similarity = calculate_ssim(original_path, screenshot_path)
            score_dict[original_image] = similarity  # 使用原始图像文件名作为键
            # print(f"已处理: {original_image} → {similarity}%")

    return score_dict


def evaluate_ssim(csv_path, image_folder, screenshot_folder):
    """主逻辑"""
    code_map = build_code_mapping(csv_path)
    score_dict = process_screenshots(screenshot_folder, image_folder, code_map)

    if score_dict:
        avg_score = sum(score_dict.values()) / len(score_dict)
        # print(f"\n有效样本数: {len(score_dict)}")
        # print(f"平均SSIM相似度: {avg_score:.1f}%")
        return avg_score, score_dict
    else:
        print("未找到有效数据对")
        return 0.0, {}


