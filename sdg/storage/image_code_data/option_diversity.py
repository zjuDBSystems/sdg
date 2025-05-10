import re
import json
import os
import time

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
        # print(f"处理 {js_path} 时出错: {e}")
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
    sil_score = silhouette_score(data, labels)
    dbi = davies_bouldin_score(data, labels)

    # 转换为 0-100 得分
    max_k = 10
    score = min(dbi * 100, 50) + min((-sil_score + 1) * 50, 50)
    return score, labels, sil_score, dbi


def find_optimal_k(data, max_k=10):
    """
    自动确定最佳聚类数 K
    :param data: 输入特征矩阵
    :param max_k: 最大尝试的 K 值
    :return: 最佳 K 值
    """
    n_samples = data.shape[0]
    if n_samples < 3:
        print("样本数量过少，无法进行聚类评估。")
        return None
    optimal_k = None
    best_score = -1
    for k in range(2, min(max_k + 1, n_samples)):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(data)
        try:
            sil_score = silhouette_score(data, labels)
            if sil_score > best_score:
                best_score = sil_score
                optimal_k = k
        except ValueError as e:
            print(e)
            continue
    return optimal_k


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
    df = build_feature_matrix(js_dir)

    # 降维可视化
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df)

    chart_type_count = find_optimal_k(reduced_data, max_k=10)
    score, labels, sil_score, dbi = calculate_diversity_score(df.values, chart_type_count)

    # 获取文件名列表
    filenames = df.index.tolist()

    # 构建簇-文件映射字典
    cluster_files = {}
    for filename, label in zip(filenames, labels):
        if label not in cluster_files:
            cluster_files[label] = []
        cluster_files[label].append(filename)

    print("========== 配置项多样性指标评估结果 ==========")
    print(f"配置项多样性最终得分: {score:.2f} 分。")
    print(f"最佳聚类数 K 值: {chart_type_count}。")
    print(f"轮廓系数 (Silhouette Score): {sil_score:.4f}。")
    print(f" Davies-Bouldin 指数 (DBI): {dbi:.4f}。")

    if score >= 80:
        print("配置项多样性非常高，簇与簇之间的平均相似性较低，进一步说明了配置项之间的差异显著且合理。")

    elif score >= 60:
        print("配置项多样性处于较好水平，表明簇与簇之间的相似性不是很高，配置项之间有一定的差异。")

    elif score >= 40:
        print("配置项多样性一般，表示簇与簇之间的平均相似性较高，配置项之间的差异不够显著，存在一定的相似性或冗余。")

    else:
        print("配置项多样性较低，表明簇与簇之间的平均相似性很高，配置项之间缺乏足够的多样性，大部分配置项较为相似。")


    print(f"\n聚类得到的簇数量: {len(cluster_files)}。")
    print(f"总字段数: {df.shape[1]}。")

    # 分析不同簇的字段分布差异
    print("\n不同簇的字段分布差异分析：")
    for cluster_id in sorted(cluster_files.keys()):
        cluster_data = df.loc[cluster_files[cluster_id]]
        cluster_fields = set(cluster_data.columns[cluster_data.sum() > 0])
        print(f"簇 {cluster_id}: 包含 {len(cluster_files[cluster_id])} 个文件，特有字段数: {len(cluster_fields)}")
        print(f"  特有字段示例: {list(cluster_fields)[:5]}...")

    # 错误报告
    # print("\n=== 错误统计 ===")
    # print(f"出错文件数: {error_count}")
    # print("典型错误文件:" + ("\n - ".join(error_files[:3]) if error_files else "无"))

        # 计算样本到簇中心的距离并输出字典
    kmeans = KMeans(n_clusters=chart_type_count, random_state=42)
    kmeans.fit(df.values)
    cluster_centers = kmeans.cluster_centers_

        # 字典：文件名 -> 到其所属簇中心的距离
    file_to_cluster_center_distance = {}
    for idx, file in enumerate(filenames):
            # 获取该文件的簇标签
        # print(file)
        cluster_label = labels[idx]
            # 计算该文件到簇中心的距离
        distance = euclidean_distances([df.iloc[idx].values], [cluster_centers[cluster_label]])[0][0]
        file_to_cluster_center_distance[file] = float(distance)
        # print(distance)

            # 打印每个文件到簇中心的距离
            # 打印每个簇的文件及其到簇中心的距离
    # print("\n每个簇的文件到簇中心的距离：")
    # for cluster_id, files in sorted(cluster_files.items()):
    #     print(f"\n簇 {cluster_id}: 包含 {len(files)} 个文件")
    #     for file in files:
    #         print(f"  文件: {file}, 距离簇中心: {file_to_cluster_center_distance[file]:.4f}")
    #         time.sleep(0.05)
    #
    #
    #     # 输出每个簇中，样本到簇中心的平均距离
    # print("\n每个簇中，各个样本到簇中心距离的平均值：")
    # cluster_avg_distances = {}
    # for cluster_id in sorted(cluster_files.keys()):
    #         # 获取该簇所有文件的索引
    #     cluster_indices = [filenames.index(file) for file in cluster_files[cluster_id]]
    #     cluster_distances = [file_to_cluster_center_distance[filenames[i]] for i in cluster_indices]
    #     avg_distance = np.mean(cluster_distances)
    #     cluster_avg_distances[cluster_id] = avg_distance
    #     print(f"簇 {cluster_id}: 平均距离 = {avg_distance:.4f}")

    # print(file_to_cluster_center_distance)
    return score,file_to_cluster_center_distance