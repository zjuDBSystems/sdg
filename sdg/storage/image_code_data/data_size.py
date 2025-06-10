import csv
import math
import os
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

    # 计算数据量对应的评分
    score = log_mapping_score(data_size, min_size, max_size)

    print("========== 数据量指标评估结果 ==========")
    print(f"文件中的条目数: {data_size}")
    print(f"数据量评分: {score:.2f} 分")

    if score == 100:
        print("数据量评分达到满分，说明CSV文件中的条目数达到或超过了设定的最大数据量 {max_size}，数据量充足，能够很好地满足相关需求。")
    elif score >= 80:
        print(f"数据量评分处于较高水平，文件中的条目数接近或超过了设定最大数据量的80%，数据量较为充足，基本能够满足使用要求。当前条目数为 {data_size}，距离最大数据量 {max_size} 还有一定差距，但已足以支撑大部分应用场景。")
    elif score >= 60:
        print(f"数据量评分处于中等水平，文件中的条目数处于设定的最小数据量 {min_size} 和最大数据量 {max_size} 之间的合理范围，数据量能够满足一般的使用需求。当前条目数为 {data_size}，虽然未达到最大数据量，但也不是过少，对于一些对数据量要求不高的任务来说是足够的。")
    elif score > 0:
        print(f"数据量评分较低，文件中的条目数相对较少，接近或低于设定最小数据量的80%，数据量可能不太充足，在一些需要大量数据的场景下可能无法很好地满足需求。当前条目数为 {data_size}，与最小数据量 {min_size} 较为接近，建议增加数据量以提高数据的可靠性和有效性。")
    else:
        print(f"数据量评分为0，说明CSV文件中的条目数小于或等于设定的最小数据量 {min_size}，数据量严重不足，可能无法满足任何实际的使用需求，需要补充大量数据。")

    return score