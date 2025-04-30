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
    chart_types = df['type'].tolist()
    entropy = calculate_shannon_entropy(chart_types)
    num_types = len(set(chart_types))
    score = calculate_score(entropy, num_types)

    print("========== 图表类型均衡性指标评估结果 ==========")
    print(f"总共有 {num_types} 种不同的图表类型。")
    print(f"计算得到的香农熵值为: {entropy:.2f}。")
    print(f"最大可能的香农熵值为: {calculate_max_entropy(num_types):.2f}。")
    print(f"图表类型均衡性最终得分: {score:.2f} 分。")
    if score == 100:
        print("图表类型分布完全均匀，各类图表的数量非常均衡，多样性达到了理想状态。")
    elif score > 70:
        print("图表类型分布较为均衡，具有较好的多样性，各类图表的数量差异不大。")
    elif score > 30:
        print("图表类型分布存在一定的不均衡，部分图表类型的数量相对较多或较少，多样性有待提高。")
    else:
        print("图表类型分布极不均衡，大部分图表集中在少数几种类型上，多样性严重不足。")

    return score

# if __name__ == '__main__':
#     CSV_FILE = "pair.csv"
#     # print(evaluate_chart_type(CSV_FILE))