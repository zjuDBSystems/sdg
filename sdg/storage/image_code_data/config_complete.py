import re
import csv
from collections import defaultdict
import json

def load_configurations(md_path):
    """加载关键配置并建立图表类型到配置的映射"""
    config_map = defaultdict(dict)

    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 分割配置块
    blocks = re.split(r'# (.*?)\n```', content)[1:]
    for i in range(0, len(blocks), 2):
        chart_type = blocks[i].strip()
        config_str = blocks[i + 1].split('```')[0].strip()

        # 转换JS对象为字典
        try:
            config_str = re.sub(r'\s+', ' ', config_str)
            # print("1")
            # print(config_str)
            config_str = re.sub(
            r'(?<=[{,])\s*(\b\w+\b)\s*(?=:)',
            r'"\1"',
            config_str,
            flags=re.DOTALL
            )
            config_str = re.sub(
                r'(:)\s*(?!(true|false)\b)(?!\d+\.?\d*)(?!\.[0-9]+)([a-zA-Z_][\w-]*)',
                lambda m: f'{m.group(1)} "{m.group(3)}"',
                config_str
            )


            config_str = re.sub(
                r'(:)\s*([-+]?\d+\.?\d*[eE][+-]?\d+)',
                lambda m: f'{m.group(1)} {m.group(2)}',
                config_str
            )
            # print("2")
            # print(config_str)
            config_str = config_str.replace("'", '"')
            # print("3")
            # print(config_str)
            config = json.loads(config_str)
            config_map[chart_type] = config
        except Exception as e:
            print(f"配置解析错误({chart_type}): {str(e)}")

    return config_map


def load_metadata(csv_path):
    """加载元数据并建立代码文件到图表类型的映射"""
    metadata = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata[row['code_file']] = row['chart_type']
    return metadata


def count_expected_items(config):
    """统计配置项总数（包含所有层级字段）"""
    count = 0

    def _traverse(node):
        nonlocal count
        if isinstance(node, dict):
            for k, v in node.items():
                count += 1  # 每个键计为1项
                _traverse(v)
        elif isinstance(node, list):
            for item in node:
                _traverse(item)
        else:
            count += 1  # 基本类型值计为1项

    _traverse(config)
    return count


def check_config_match(js_code, expected_config):
    """检查JS代码与配置的匹配度"""
    match_count = 0

    def _check_fields(node, parent_keys=[]):
        nonlocal match_count
        if isinstance(node, dict):
            for key, value in node.items():
                current_path = '.'.join(parent_keys + [key])

                # 检查字段存在性
                pattern = rf'\b{re.escape(key)}\s*:'
                if re.search(pattern, js_code):
                    match_count += 1
                    # print(f"字段存在: {current_path}")

                # 递归检查子配置
                _check_fields(value, parent_keys + [key])
        elif isinstance(node, list):
            for item in node:
                _check_fields(item, parent_keys)
        else:
            # 检查值匹配
            if parent_keys:
                parent_key = parent_keys[-1]
                pattern = rf'{re.escape(parent_key)}\s*:\s*([\'"]?){re.escape(str(node))}\1'
                if re.search(pattern, js_code):
                    match_count += 1
                    # print(f"值匹配: {'.'.join(parent_keys)} = {node}")

    _check_fields(expected_config)
    return match_count


def process_files(config_map, metadata, js_dir) -> dict:
    """处理所有JS文件，返回 {代码文件: 得分} 字典"""
    file_scores = {}

    for code_file, chart_type in metadata.items():
        # 初始化默认得分
        score = 0.0

        # 检查配置是否存在
        if chart_type not in config_map:
            print(f"警告：未找到{chart_type}的配置")
        else:
            try:
                # 读取JS代码
                with open(f"{js_dir}/{code_file}", 'r', encoding='utf-8') as f:
                    js_code = f.read()

                # 计算匹配度
                expected_config = config_map[chart_type]
                total_items = count_expected_items(expected_config)
                if total_items > 0:
                    matched = check_config_match(js_code, expected_config)
                    score = (matched / total_items) * 100
            except FileNotFoundError:
                print(f"文件不存在：{code_file}")

        # 记录得分（保留1位小数）
        file_scores[code_file] = round(score, 1)

    return file_scores






def evaluate_completeness(config_path,pair_path,js_dir):
    config_map = load_configurations(config_path)
    metadata = load_metadata(pair_path)

    # 获取得分字典
    file_scores = process_files(config_map, metadata, js_dir)

    # 计算平均分
    valid_scores = [s for s in file_scores.values() if s > 0]
    avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    # 输出结果
    # print(f"\n配置完整性平均分: {avg_score:.1f}%")
    return avg_score, file_scores


