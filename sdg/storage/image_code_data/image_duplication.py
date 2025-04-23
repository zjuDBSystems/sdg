import os
from PIL import Image
import imagehash


def calculate_phash(image_path, hash_size=16):
    """使用imagehash库生成感知哈希"""
    try:
        with Image.open(image_path) as img:
            # 保持与原始方法一致的参数：hash_size控制DCT后的矩阵大小
            return imagehash.phash(img, hash_size=hash_size)
    except Exception as e:
        print(f"无法处理图像 {image_path}: {str(e)}")
        return None


def find_duplicate_images(image_dir, threshold=5):
    """改进后的重复检测方法（返回总样本数）"""
    hash_dict = {}  # {hash: [文件列表]}
    duplicates = set()
    total_processed = 0  # 新增计数器

    # 第一遍：批量生成哈希
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        img_hash = calculate_phash(img_path)
        if img_hash is not None:
            total_processed += 1
            hash_str = str(img_hash)
            if hash_str not in hash_dict:
                hash_dict[hash_str] = []
            hash_dict[hash_str].append(img_file)

    # 第二遍：在相似哈希组内比较
    for group in hash_dict.values():
        if len(group) < 2:
            continue

        # 组内两两比较
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                file1 = os.path.join(image_dir, group[i])
                file2 = os.path.join(image_dir, group[j])

                hash1 = calculate_phash(file1)
                hash2 = calculate_phash(file2)

                if hash1 - hash2 <= threshold:
                    duplicates.add(group[i])
                    duplicates.add(group[j])

    return duplicates, total_processed  # 返回实际处理的图片总数


# 以下函数保持原有逻辑不变
def calculate_quality_score(duplicates, total_images):
    if total_images == 0:
        return 100.0  # 空数据集视为完美质量

        # 计算不重复样本数 = 总样本数 - 重复样本数
    unique_count = total_images - len(duplicates)
    return max(0.0, min(100.0, (unique_count / total_images) * 100))


def evaluate_image_duplicate(dataset_path):
    duplicates, total = find_duplicate_images(dataset_path)
    score=calculate_quality_score(duplicates, total)
    return score, duplicates