import re
import json
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
error_count = 0
error_files = []


# --------------------------
# 核心修改部分：字段存在性特征提取
# --------------------------
def extract_field_names(js_path):
    """提取option对象中的所有顶级字段名"""
    global error_count, error_files
    try:
        with open(js_path, 'r', encoding='utf-8') as f:
            js_data = json.load(f)

        fields = set()
        def traverse_json(data):
            if isinstance(data, dict):
                for key in data.keys():
                    fields.add(key)
                    traverse_json(data[key])
            elif isinstance(data, list):
                for item in data:
                    traverse_json(item)

        traverse_json(js_data)

        return list(fields)

    except Exception as e:
        print(f"处理 {js_path} 时出错: {e}")
        error_count += 1
        error_files.append(js_path)
        return []



def build_feature_matrix(js_dir):
    """构建字段存在性特征矩阵"""
    # 第一阶段：收集所有可能的字段
    all_fields = set()
    file_records = []

    for root, _, files in os.walk(js_dir):
        for file in files:
            if file.endswith('.json'):
                js_path = os.path.join(root, file)
                fields = extract_field_names(js_path)
                file_records.append((file, fields))
                all_fields.update(fields)

    # 转换为排序的字段列表，确保列顺序一致
    field_columns = sorted(all_fields)

    # 第二阶段：构建特征矩阵
    feature_rows = []
    filenames = []

    for file, fields in file_records:
        # 生成存在性特征向量（1表示存在，0不存在）
        row = {field: 1 for field in fields}
        feature_row = [row.get(col, 0) for col in field_columns]
        feature_rows.append(feature_row)
        filenames.append(file)

    return pd.DataFrame(feature_rows, columns=field_columns, index=filenames)





def calculate_diversity_score(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    #
    # # 计算 DBI
    dbi = davies_bouldin_score(data, labels)
    # print("dbi", dbi)
    # 转换为 0-100 得分
    max_k = 10
    score = min(dbi * 180, 70) + (n_clusters / max_k) * 30
    return score, labels


def find_optimal_k(data, max_k=10):
    """
    自动确定最佳聚类数 K
    :param data: 输入特征矩阵
    :param max_k: 最大尝试的 K 值
    :return: 最佳 K 值
    """
    silhouette_scores = []

    # 尝试 K 从 2 到 max_k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)

        # 计算轮廓系数（仅当有至少两个簇）
        if len(np.unique(labels)) >= 2:
            sil_score = silhouette_score(data, labels)
        else:
            sil_score = -1  # 无效值
        silhouette_scores.append(sil_score)
    # 自动选择最佳 K
    # 策略：轮廓系数最高且 SSE 下降平缓
    best_k_sil = np.argmax(silhouette_scores) + 2  # +2 因为从k=2开始

    # print(best_k_sil)


    return best_k_sil


def _find_elbow_point(sse):
    """
    通过曲率计算肘部点
    :param sse: 每个 K 对应的 SSE 值列表
    :return: 肘部点索引（从0开始）
    """
    # 计算二阶导数找拐点
    second_deriv = np.diff(np.diff(sse))
    if len(second_deriv) == 0:
        return 0
    return np.argmax(second_deriv)



def evaluate_option_diversity(js_dir, csv_path):
    # 构建二进制特征矩阵
    # 构建二进制特征矩阵
    df = build_feature_matrix(js_dir)

    # print("\n=== 特征维度分析 ===")
    # print(f"总字段数: {df.shape[1]}")
    # print(f"示例字段: {df.columns.tolist()[:5]}...")

    # 降维可视化
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)

    # 计算得分（假设有5种图表类型）
    # chart_type_count = 3
    chart_type_count = find_optimal_k(df.values, max_k=10)
    # print("最佳k值",chart_type_count)
    score, labels = calculate_diversity_score(reduced_data, chart_type_count)
    # print(score)
    # 获取文件名列表
    filenames = df.index.tolist()

    # 构建簇-文件映射字典
    cluster_files = {}
    for filename, label in zip(filenames, labels):
        if label not in cluster_files:
            cluster_files[label] = []
        cluster_files[label].append(filename)

    # # 打印聚类结果
    # print("\n=== 聚类结果分布 ===")
    # for cluster_id in sorted(cluster_files.keys()):
    #     print(f"簇 {cluster_id}: {len(cluster_files[cluster_id])} 个文件")
    #
    # # 详细输出每个簇的文件列表
    # print("\n=== 详细文件分布 ===")
    # for cluster_id, files in cluster_files.items():
    #     print(f"\n簇 {cluster_id} 包含以下 {len(files)} 个文件：")
    #     for idx, filename in enumerate(files, 1):
    #         print(f"{idx}. {filename}")
    #
    # # 保存结果到文件
    # with open("cluster_results.txt", "w", encoding="utf-8") as f:
    #     for cluster_id in sorted(cluster_files.keys()):
    #         f.write(f"=== 簇 {cluster_id} ===\n")
    #         for filename in cluster_files[cluster_id]:
    #             f.write(f"{filename}\n")
    #         f.write("\n")
    #
    # # # 可视化（保持不变）
    # plt.figure(figsize=(10, 6))
    # plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='tab20', alpha=0.7)
    # plt.title(f"配置项字段聚类 (多样性评分: {score:.1f}/100)")
    # plt.xlabel("主成分1")
    # plt.ylabel("主成分2")
    # plt.show()

    # 错误报告（保持不变）
    # print("\n=== 错误统计 ===")
    # print(f"出错文件数: {error_count}")
    # print("典型错误文件:" + ("\n - ".join(error_files[:3]) if error_files else "无"))
    return score

