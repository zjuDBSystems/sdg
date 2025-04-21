import pandas as pd
import math
from collections import Counter


def calculate_shannon_entropy(chart_types):
    """计算Shannon Entropy（香农熵）来衡量多样性"""
    if len(chart_types) == 0:
        return 0
    count = Counter(chart_types)
    total = len(chart_types)
    probabilities = [count[type_] / total for type_ in count]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy


def calculate_max_entropy(num_types):
    """计算最大熵，最大熵对应于均匀分布"""
    return math.log2(num_types)


def calculate_score(entropy, num_types):
    """将Shannon Entropy值转换为0-100分，动态确定最大熵"""
    max_entropy = calculate_max_entropy(num_types)
    score = max(0, min(100, (entropy / max_entropy) * 100))
    return score


def evaluate_chart_type(csv_path):
    df = pd.read_csv(csv_path)
    chart_types = df['chart_type'].tolist()
    # print(chart_types)
    entropy = calculate_shannon_entropy(chart_types)
    num_types = len(set(chart_types))
    score = calculate_score(entropy, num_types)
    return score

# if __name__ == '__main__':
#     CSV_FILE = "pair.csv"
#     print(evaluate_chart_type(CSV_FILE))