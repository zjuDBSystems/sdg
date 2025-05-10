
import pandas as pd

def evaluate_miss(metadata_path):
    # 读取CSV文件
    df = pd.read_csv(metadata_path)

    # 检查每一行是否有缺失值
    missing_data_count = df.isnull().any(axis=1).sum()
    non_missing_data_count = len(df) - missing_data_count
    completeness_score = (non_missing_data_count / len(df)) * 100

    # 创建缺失列名字典
    missing_dict = {}
    for index, row in df.iterrows():
        missing_cols = []
        for col in df.columns:
            if pd.isnull(row[col]):
                missing_cols.append(col)
        if missing_cols:
            missing_dict[index] = missing_cols

    # 打印评估结果
    print("========== 缺失率指标评估结果 ==========")
    print(f"数据集总行数: {len(df)}")
    print(f"包含缺失值的行数: {missing_data_count}")
    print(f"不包含缺失值的行数: {non_missing_data_count}")
    print(f"数据完整性分数（百分比）: {completeness_score:.2f}%")

    # 打印缺失详情
    print("\n缺失详情：")
    if missing_dict:
        for index, cols in missing_dict.items():
            print(f"行 {index + 1}: 缺失列 - {', '.join(cols)}")
    else:
        print("无缺失数据。")

    # 根据分数给出建议
    if completeness_score == 100:
        print("\n所有数据行均无缺失值，数据完整性非常高，无需进行缺失值处理。")
    elif completeness_score > 70:
        print("\n大部分数据行无缺失值，数据完整性较好，但仍需处理部分缺失值。")
    else:
        print("\n数据中存在较多缺失值，建议进行填补或删除处理。")

    return completeness_score, missing_dict
