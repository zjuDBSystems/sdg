import numpy as np
from sklearn.neural_network import MLPRegressor


# Step 1: 模拟Deepseek接口返回结果（实际应调用API）
def get_llm_analysis():

    return {
        "top_5_secondary_metrics": [
            "配置项多样性",  # 权重1.0（多样性核心指标[[3]]）
            "图像与渲染截图的SSIM",  # 权重0.8（对齐质量关键[[2]]）
            "数据量",  # 权重0.7（基础要素[[4]]）
            "联合重复",  # 权重0.5（重复检测核心[[4]]）
            "类型均衡性"  # 权重0.4（多样性扩展[[10]]）
        ]
    }


# Step 2: 构建MLP模型（预训练权重假设已存在）
class QualityPredictor:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(20, 10), max_iter=500)
        # 假设已用标准化数据训练，输入维度为10个二级指标

    def predict(self, X):
        # X是形状为(10,)的数组，包含所有二级指标评分
        return self.model.predict([X])


# Step 3: 加权计算主函数
def calculate_top_metrics(dataset_quality_scores):
    """
    :param dataset_quality_scores: 字典，包含所有10个二级指标的评分
    """
    # 获取LLM分析结果
    llm_result = get_llm_analysis()
    top_metrics = llm_result["top_5_secondary_metrics"]
    weights = [1.0, 0.8, 0.7, 0.5, 0.4]

    # 准备神经网络输入
    all_metrics = [
        "语法检测", "可渲染性检测", "配置项完整检测",
        "图像与渲染截图的SSIM", "图像OCR检测的文字与配置项的余弦相似度",
        "图表类型均衡性", "配置项多样性",
        "代码重复", "图像重复", "联合重复",
        "数据量"
    ]
    X = [dataset_quality_scores[metric] for metric in all_metrics]

    # 获取神经网络预测值
    predictor = QualityPredictor()
    nn_scores = predictor.predict(X)

    # 创建指标-得分映射
    metric_score_map = {metric: score for metric, score in zip(all_metrics, nn_scores)}

    # 计算加权分数
    weighted_scores = {}
    for metric, weight in zip(top_metrics, weights):
        weighted_scores[metric] = metric_score_map[metric] * weight

    # 排序并返回前三
    sorted_metrics = sorted(weighted_scores.items(), key=lambda x: -x[1])
    return [metric for metric, _ in sorted_metrics[:3]]


# 使用示例
sample_scores = {
    "语法检测": 0.85,
    "可渲染性检测": 0.92,
    "配置项完整检测": 0.78,
    "图像与渲染截图的SSIM": 0.88,
    "图像OCR检测的文字与配置项的余弦相似度": 0.81,
    "图表类型均衡性": 0.75,
    "配置项多样性": 0.93,
    "代码重复": 0.68,
    "图像重复": 0.72,
    "联合重复": 0.80,
    "数据量": 0.89
}

print(calculate_top_metrics(sample_scores))  # 输出前三维度
