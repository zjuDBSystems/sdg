import pandas as pd


def calculate_score_from_csv(csv_path,
                    row_threshold=20000, col_threshold=50):
    """返回 {'行数规模':row_score, '列数规模':col_score} 字典"""
    df = pd.read_csv(csv_path)
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes("number").shape[1]

    row_score = min(n_rows / row_threshold, 1) * 100
    col_score = min(num_cols / col_threshold, 1) * 100
    return row_score, col_score
