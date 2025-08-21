from typing import List, Sequence

import numpy as np
import pandas as pd

from sdg.storage.power_table_data import _distance_corr


def score_feature_independence(
        df_list: List[pd.DataFrame],
        target_col: str,
        remain_cols: Sequence[str] | None = None,
        method: str = 'pearson'  # 可选 'pearson' 或 'spearman'
) -> float:
    remain_cols = list(remain_cols or [])
    ffi_vals, tfi_vals, weights = [], [], []

    for df in df_list:
        df_filled = df.fillna(0)

        # --- 2.1 选出要评估的特征列 ---
        # 从填充过的副本中选择特征
        feats = df_filled[remain_cols].select_dtypes("number")

        d = feats.shape[1]
        if d < 2:
            continue

        # --- 2.2 计算 FFI ---
        corr_matrix = feats.corr(method=method).abs()
        np.fill_diagonal(corr_matrix.values, 0)
        avg_corr = corr_matrix.sum().sum() / (d * (d - 1))

        ffi = 1 - avg_corr

        # --- 2.3 计算 TFI ---
        try:
            tfi = 1 - _distance_corr(df_filled[target_col], feats.mean(axis=1))
        except NameError:
            tfi = ffi

        ffi_vals.append(ffi)
        tfi_vals.append(tfi)
        weights.append(len(df))

    if not ffi_vals:
        return 100.0

    # --- 2.4 样本量加权平均 ---
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    ffi_avg = float(np.dot(ffi_vals, w))
    tfi_avg = float(np.dot(tfi_vals, w))

    return 0.5 * (ffi_avg + tfi_avg) * 100