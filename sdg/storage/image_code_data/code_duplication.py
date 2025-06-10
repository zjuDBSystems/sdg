import json
import hashlib
import os
import difflib
from collections import defaultdict


def normalize_json(json_data):
    """标准化JSON数据"""
    return json.dumps(json_data, sort_keys=True)


def calculate_hash(json_str):
    """计算标准化JSON字符串的哈希值"""
    return hashlib.md5(json_str.encode('utf-8')).hexdigest()


def calculate_similarity(json_str1, json_str2):
    """计算两个标准化JSON字符串的相似度"""
    return difflib.SequenceMatcher(None, json_str1, json_str2).ratio()


def process_dataset(dataset_path):
    """遍历数据集中的所有JSON文件并计算相似度"""
    all_code = []
    all_json = []
    json_files = []

    # 遍历指定路径下的所有文件
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):  # 只处理 JSON 文件
                json_code_path = os.path.join(root, file)
                json_files.append(file)  # 存储文件名

                # 读取代码并解析为JSON
                with open(json_code_path, 'r', encoding='utf-8') as f:
                    try:
                        code = f.read()
                        json_data = json.loads(code)
                        normalized_json = normalize_json(json_data)  # 标准化JSON
                        all_code.append(code)
                        all_json.append(normalized_json)  # 存储标准化后的JSON
                    except json.JSONDecodeError as e:
                        # print(f"Error decoding JSON file {json_code_path}: {e}")
                        continue

    return json_files, all_code, all_json


def calculate_duplicate_rate(all_json, json_files):
    """计算代码的重复率，并返回重复代码的文件名"""
    hash_groups = defaultdict(list)

    # 生成哈希指纹
    for idx, json_str in enumerate(all_json):
        hashed = calculate_hash(json_str)  # 计算哈希值
        hash_groups[hashed].append(idx)

    duplicate_count = 0
    duplicate_files = set()

    # 只在同哈希组内比较
    for group in hash_groups.values():
        if len(group) < 2:
            continue

        # 组内两两比较
        for i in range(len(group)):
            idx1 = group[i]
            duplicate_count += 1
            duplicate_files.add(json_files[idx1])

    duplicate_files = list(duplicate_files)
    total_pairs = len(all_json) * (len(all_json) - 1) / 2
    duplicate_rate = duplicate_count / len(all_json)

    return duplicate_rate, duplicate_files


def calculate_quality_score(duplicate_rate):
    """将重复率转换为质量得分"""
    score = max(0, min(100, (1 - duplicate_rate) * 100))  # 重复率越高，得分越低
    return score


def evaluate_code_duplicate(dataset_path):
    json_files, all_code, all_json = process_dataset(dataset_path)
    duplicate_rate, duplicate_files = calculate_duplicate_rate(all_json, json_files)
    quality_score = calculate_quality_score(duplicate_rate)

    print("========== 代码重复率指标评估结果 ==========")
    print(f"代码重复率: {duplicate_rate * 100:.2f}%")
    print(f"代码质量得分: {quality_score} 分")

    if quality_score >= 90:
        print(f"代码质量非常高，代码重复率很低，说明代码的编写具有较高的独立性和原创性，重复的代码很少。")
    elif quality_score >= 60:
        print(f"代码质量处于较好水平，代码重复率相对较低，代码之间的重复部分不是很多，整体较为良好。")
    elif quality_score >= 40:
        print(f"代码质量一般，代码重复率适中，存在一定量的重复代码，可能会影响代码的维护性和可读性。")
    else:
        print(f"代码质量较低，代码重复率较高，大量的代码存在重复，这会增加代码的维护成本，降低代码的质量。")

    if duplicate_files:
        print(f"\n存在重复的文件有: {duplicate_files}")
    # else:
    #     print("\n没有发现重复的文件。")

    return quality_score, duplicate_files