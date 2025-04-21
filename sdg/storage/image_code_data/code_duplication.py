import ast
import difflib
import os
import numpy as np
import re
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
        code_no_comments = remove_comments(code)
        js_code = re.sub(r'//.*', '', code)  # 移除单行注释
        js_code = re.sub(r'/\*.*?\*/', '', js_code, flags=re.DOTALL)  # 移除多行注释
        # 匹配 option 对象
        # 匹配所有以 option 开头的配置项，如 option1, option2 等
        option_match = re.search(r'option\s*=\s*({.*?});', js_code, re.DOTALL)
        # print(option_match)
        option_code=option_match.group(1)
        # print(option_code)
        return ast.parse(option_code)
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
            if file.endswith(".js"):  # 只处理 JS 文件
                js_code_path = os.path.join(root, file)
                js_files.append(file)  # 存储文件名
                # print(f"Processing: {js_code_path}")

                # 读取代码并解析为AST
                with open(js_code_path, 'r', encoding='utf-8') as f:
                    code = f.read()

                # 将代码解析为AST
                ast_node = parse_code_to_ast(code)
                if ast_node:
                    all_code.append(code)
                    all_asts.append(ast_node)

    return js_files, all_code, all_asts

def calculate_duplicate_rate(all_asts, js_files, threshold=0.8):
    """计算代码的重复率，并返回重复代码的文件名"""
    total_pairs = len(all_asts) * (len(all_asts) - 1) / 2
    duplicate_count = 0
    duplicate_files = []  # 用于存储重复的代码文件名

    # 比较每对代码的AST相似度
    for i in range(len(all_asts)):
        for j in range(i + 1, len(all_asts)):
            similarity = calculate_similarity(all_asts[i], all_asts[j])
            if similarity > threshold:  # 设置相似度阈值为 0.8，表示重复
                duplicate_files.append(js_files[i])
                duplicate_files.append(js_files[j])
                duplicate_count += 1

    # 去除重复的文件名
    duplicate_files = list(set(duplicate_files))

    # 计算重复率
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



