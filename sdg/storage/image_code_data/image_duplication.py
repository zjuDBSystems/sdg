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
    score = calculate_quality_score(duplicates, total)

    print("========== 图像重复指标评估结果 ==========")
    print(f"处理的图像总数: {total} 张")
    print(f"图像质量得分: {score:.2f} 分")

    if score >= 90:
        print(f"图像质量非常高，图像重复率很低，说明图像数据集的多样性较好，重复的图像对整体影响较小。")
    elif score >= 60:
        print(f"图像质量处于较好水平，图像重复率相对较低，但不影响整体的图像质量，图像之间的重复部分不是很多。")
    elif score >= 40:
        print(f"图像质量一般，图像重复率适中，这可能会对图像数据集的多样性和后续使用产生一定影响。")
    else:
        print(f"图像质量较低，图像重复率较高，这大大降低了图像数据集的质量，可能会影响到基于这些图像的分析和应用。")

    if duplicates:
        print(f"\n存在重复的图像文件有: {duplicates}")
    # else:
    #     print("\n没有发现重复的图像文件。")

    return score, duplicates