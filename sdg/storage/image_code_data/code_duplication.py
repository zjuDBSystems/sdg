import ast
import difflib
import os
import numpy as np
import re
from hashlib import md5
from collections import defaultdict
def remove_comments(code):
    """删除代码中的注释"""
    # 移除单行注释
    code = re.sub(r'//.*', '', code)
    # 移除多行注释
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code




def parse_code_to_ast(code):
    """将代码解析为AST"""
    # print(code)
    try:
        if code.strip().startswith("{") and code.strip().endswith("}"):
            code = "option = " + code

        return ast.parse(code)
    except SyntaxError as e:
        print("存在语法错误")
        print(e)
        return None

def ast_to_str(node):
    """将AST节点转换为标准化的字符串"""
    return ast.dump(node, annotate_fields=False)

def calculate_similarity(ast1, ast2):
    """计算两棵AST树的相似度"""
    # 将AST转换为字符串
    ast1_str = ast_to_str(ast1)
    ast2_str = ast_to_str(ast2)

    # 使用difflib计算字符串的相似度
    similarity = difflib.SequenceMatcher(None, ast1_str, ast2_str).ratio()
    return similarity

def process_dataset(dataset_path):
    """遍历数据集中的所有JS配置文件并计算相似度"""
    all_code = []
    all_asts = []
    js_files = []
    # 遍历指定路径下的所有文件
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".json"):  # 只处理 JS 文件
                js_code_path = os.path.join(root, file)
                js_files.append(file)  # 存储文件名
                # print(f"Processing: {js_code_path}")

                # 读取代码并解析为AST
                with open(js_code_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                # print(js_code_path)
                # 将代码解析为AST
                ast_node = parse_code_to_ast(code)
                # print(ast_node)
                if ast_node:
                    all_code.append(code)
                    all_asts.append(ast_node)

    return js_files, all_code, all_asts

def calculate_duplicate_rate(all_asts, js_files, threshold=0.8):
    """计算代码的重复率，并返回重复代码的文件名"""
    hash_groups = defaultdict(list)

    # 生成哈希指纹
    for idx, ast_node in enumerate(all_asts):
        ast_str = ast_to_str(ast_node)
        # 生成128位MD5哈希
        hashed = md5(ast_str.encode()).hexdigest()
        hash_groups[hashed].append(idx)

    duplicate_count = 0
    duplicate_files = set()

    # 只在同哈希组内比较
    for group in hash_groups.values():
        if len(group) < 2:
            continue

        # 组内两两比较
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                idx1, idx2 = group[i], group[j]
                similarity = calculate_similarity(all_asts[idx1], all_asts[idx2])

                if similarity > threshold:
                    duplicate_count += 1
                    duplicate_files.add(js_files[idx1])
                    duplicate_files.add(js_files[idx2])

    # 后续处理保持不变
    duplicate_files = list(duplicate_files)
    total_pairs = len(all_asts) * (len(all_asts) - 1) / 2
    duplicate_rate = duplicate_count / total_pairs if total_pairs > 0 else 0

    return duplicate_rate, duplicate_files


def calculate_quality_score(duplicate_rate):
    """将重复率转换为质量得分"""
    score = max(0, min(100, (1 - duplicate_rate) * 100))  # 重复率越高，得分越低
    return score

def evaluate_code_duplicate(dataset_path):
    js_files, all_code, all_asts = process_dataset(dataset_path)
    duplicate_rate, duplicate_files = calculate_duplicate_rate(all_asts, js_files)
    quality_score = calculate_quality_score(duplicate_rate)
    return quality_score,duplicate_files



