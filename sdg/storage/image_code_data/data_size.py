import csv
import math
import os
import chardet
import pandas as pd


def log_mapping_score(data_size: int, min_size: int, max_size: int) -> float:
    """
    对数映射评分（0-100分）
    :param data_size: 当前数据量
    :param min_size: 最小数据量（得0分）
    :param max_size: 最大数据量（得100分）
    :return: 标准化得分
    """
    if data_size <= min_size:
        return 0.0
    elif data_size >= max_size:
        return 100.0
    else:
        # 避免对0或负数取对数
        numerator = math.log(data_size + 1) - math.log(min_size + 1)
        denominator = math.log(max_size + 1) - math.log(min_size + 1)
        return (numerator / denominator) * 100




def calculate_score_from_csv(csv_file: str, min_size: int, max_size: int) -> float:
    """从CSV文件中获取配对条目数并计算分数"""
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 获取条目数
    data_size = len(df)
    # print(data_size)

    # 计算数据量对应的评分
    score = log_mapping_score(data_size, min_size, max_size)

    return score



