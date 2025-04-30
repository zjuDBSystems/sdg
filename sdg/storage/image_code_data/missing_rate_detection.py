import pandas as pd


def evaluate_miss(metadata_path):
    # 读取CSV文件
    df = pd.read_csv(metadata_path)

    # 检查每一行是否有任何缺失值
    missing_data_count = df.isnull().any(axis=1).sum()  # 统计有缺失的行数

    # 计算不缺失行数
    non_missing_data_count = len(df) - missing_data_count
    # print(non_missing_data_count)
    # 计算完整性分数
    completeness_score = (non_missing_data_count / len(df)) * 100

    print("========== 缺失率指标评估结果 ==========")
    print(f"数据集总行数: {len(df)}")
    print(f"包含缺失值的行数: {missing_data_count}")
    print(f"不包含缺失值的行数: {non_missing_data_count}")
    print(f"数据完整性分数（百分比）: {completeness_score:.2f}%")

    if completeness_score == 100:
        print("所有数据行均无缺失值，数据完整性非常高，说明该数据集在数据记录上较为完整，无需进行缺失值处理。")
    elif completeness_score > 70:
        print("大部分数据行无缺失值，数据完整性较好，但仍存在一些缺失值，可能需要根据具体情况决定是否对缺失值进行插补或删除等处理。")
    else:
        print("数据中存在较多缺失值，数据完整性较低，缺失值可能会对后续的数据分析和模型训练产生较大影响，建议对缺失值进行填补或对相关数据行进行处理。")

    return completeness_score
