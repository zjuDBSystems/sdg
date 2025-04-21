import os
import cv2
import numpy as np
from PIL import Image

def phash(image_path, hash_size=16):
    """计算图像的感知哈希"""
    img = Image.open(image_path).convert("L").resize((hash_size, hash_size))
    pixels = np.array(img)
    dct_coeffs = cv2.dct(pixels.astype(np.float32))
    dct_coeffs_roi = dct_coeffs[:8, :8]  # 取低频部分
    avg = np.mean(dct_coeffs_roi)
    hash_str = ''.join(['1' if x > avg else '0' for x in dct_coeffs_roi.flatten()])
    return hash_str

def hamming_distance(hash1, hash2):
    """计算两个二进制哈希字符串的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
def find_duplicate_images(image_dir, threshold=5):
    """查找重复的图像，使用阈值判断重复"""
    hashes = {}
    duplicates = set()

    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img_hash = phash(img_path)

        # 比较当前图像的哈希值与已有哈希值的汉明距离
        for existing_hash, existing_file in hashes.items():
            if hamming_distance(img_hash, existing_hash) <= threshold:
                # duplicates.append((img_file, existing_file))
                duplicates.add(img_file)
                duplicates.add(existing_file)
                break
        else:
            hashes[img_hash] = img_file

    return duplicates, len(hashes)

def calculate_quality_score(duplicates, total_images):
    """根据重复图像计算质量得分"""
    duplicates_num=len(duplicates)/2
    duplicate_rate = duplicates_num / total_images if total_images > 0 else 0
    score = max(0, min(100, (1 - duplicate_rate) * 100))
    return score

def evaluate_image_duplicate(dataset_path):
    image_duplicates, total_images = find_duplicate_images(dataset_path)

    # 计算质量得分
    quality_score = calculate_quality_score(image_duplicates, total_images)
    return quality_score,image_duplicates



