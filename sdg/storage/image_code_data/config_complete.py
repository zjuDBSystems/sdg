import json
import math
import re
from collections import Counter
import os

import pandas as pd


def load_configurations(md_path):
    configurations = []
    with open(md_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith('#'):
                name = lines[i].strip().lstrip('# ').strip()
                i += 1
                while i < len(lines) and not lines[i].startswith('```'):
                    i += 1
                i += 1
                config = ''
                while i < len(lines) and not lines[i].startswith('```'):
                    config += lines[i].strip()
                    i += 1
                config = config.replace("'", "\"")
                config = re.sub(r'(\w+):', r'"\1":', config)
                config = re.sub(r':\s*true', ': true', config)
                config = re.sub(r':\s*false', ': false', config)
                config = re.sub(r':\s*([^",\{\}\[\]]+)(?=\s*[,\}\]])', r': "\1"', config)
                try:
                    config = json.loads(config)
                    configurations.append({'name': name, 'config': config})
                except json.JSONDecodeError:
                    print(f"配置 {name} 解析错误，请检查格式，错误内容：{config}")
            else:
                i += 1
    return configurations


def extract_option_from_js(js_path):
    try:
        with open(js_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
        js_code = js_code.replace("'", "\"")
        js_code = re.sub(r'(\w+):', r'"\1":', js_code)
        js_code = re.sub(r'(\b(?:[a-zA-Z0-9_]+)\b):\s*\'([^\']+)\'', r'"\1": "\2"', js_code)
        js_code = re.sub(r'\/\/[^\n]*', '', js_code)
        option_start = js_code.find('option = ') + len('option = ')
        option_end = js_code.find(';', option_start)
        option_code = js_code[option_start:option_end].strip()
        return json.loads(option_code)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"处理 {js_path} 时出错: {e}")
        return {}


# def count_config_items(config):
#     count = 0
#     if isinstance(config, dict):
#         for key, value in config.items():
#             count += 1
#             if isinstance(value, dict):
#                 count += count_config_items(value)
#     elif isinstance(config, list):
#         for item in config:
#             if isinstance(item, dict):
#                 count += count_config_items(item)
#     return count

def count_config_items(config):
    """统计目标配置的总项数（键和基本类型值均为独立项）"""
    count = 0
    if isinstance(config, dict):
        for key, value in config.items():
            count += 1  # 键本身计为1项
            count += count_config_items(value)  # 递归处理值
    elif isinstance(config, list):
        for item in config:
            count += count_config_items(item)
    else:
        count += 1  # 基本类型值计为1项
    return count


def match_config(js_config, target_config):
    """递归匹配配置项并返回匹配的数量"""
    match_count = 0

    # 处理字典类型配置
    if isinstance(target_config, dict):
        if isinstance(js_config, list):
            # 如果js配置是数组，遍历每个元素尝试匹配
            for item in js_config:
                match_count += match_config(item, target_config)
            return match_count
        elif not isinstance(js_config, dict):
            return 0

        # 遍历目标配置的所有键
        for key in target_config:
            if key in js_config:
                match_count += 1  # 键匹配成功
                # print(key)
                # 递归处理值
                match_count += match_config(js_config[key], target_config[key])

    # 处理列表类型配置（按顺序严格匹配）
    elif isinstance(target_config, list):
        if not isinstance(js_config, list):
            return 0
        # 逐个匹配列表元素
        for t_item, j_item in zip(target_config, js_config):
            match_count += match_config(j_item, t_item)
            # print(j_item,t_item)

    # 处理基本类型（值必须严格相等）
    else:
        str_js = str(js_config).lower() if isinstance(js_config, bool) else str(js_config)
        str_target = str(target_config).lower() if isinstance(target_config, bool) else str(target_config)
        if str_js == str_target:
            match_count += 1

    return match_count



def check_config_completeness(js_option, target_config):
    target_count = count_config_items(target_config)
    # print(target_config)
    # print(target_count)
    # print(js_option)
    match_count = match_config(js_option, target_config)
    # print(match_count)
    score = (match_count / target_count) * 100 if target_count > 0 else 0
    return score


# def classify_js_config(js_option, configurations):
#     for config in configurations:
#         if match_config(js_option, config['config']):
#             return config['name']
#     return '其他类别'


def evaluate_completeness( md_path, csv_path,js_dir):
    """返回元组 (平均分, {代码文件名: 得分})"""
    configurations = load_configurations(md_path)
    # print(configurations)
    df = pd.read_csv(csv_path)
    score_dict = {}  # 新增：存储文件名和得分的字典
    total_score = 0

    for root, _, files in os.walk(js_dir):
        for file in files:
            if file.endswith('.json'):
                js_path = os.path.join(root, file)
                try:
                    # 加载JS配置
                    with open(js_path, 'r', encoding='utf-8') as f:
                        js_option = json.load(f)

                    # 获取对应的图表类型
                    row = df[df['code'] == file]
                    if not row.empty:
                        chart_type = row['type'].values[0]
                        # 查找匹配的配置
                        for config in configurations:
                            if config['name'] == chart_type:
                                # 计算得分


                                score = check_config_completeness(js_option, config['config'])
                                # if(score>100 or score<100):
                                #     print(js_path)
                                #     print(score)
                                score_dict[file] = score  # 记录到字典
                                total_score += score
                                break
                        else:
                            print(f"警告：未找到{chart_type}的配置")
                            score_dict[file] = 0.0
                    else:
                        print(f"未在CSV中找到{file}对应记录")
                        score_dict[file] = 0.0
                except Exception as e:
                    print(f"处理 {file} 出错: {str(e)}")
                    score_dict[file] = 0.0

    # 计算平均分
    avg_score = total_score / len(score_dict) if score_dict else 0.0
    return avg_score, score_dict


# def main():
#     # 请根据实际情况修改这些路径
#     js_dir = './monishujuji'
#     md_path = '关键配置.md'
#     csv_path = 'pair.csv'
#
#     scores,average_score = evaluate_completeness(js_dir, md_path, csv_path)
#
#
#
#     print("\n配置项完整性得分:")
#     for i, score in enumerate(scores, 1):
#         print(f"文件 {i}: {score} 分")
#     print(average_score)
#
#
# if __name__ == "__main__":
#     main()
