from math import e
from typing import List, Sequence, Literal, Optional

import numba
import numpy as np
import pandas as pd
from scipy.signal import periodogram
from scipy.stats import entropy, skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def _concat_series(
        df_list: List[pd.DataFrame],
        col: str | int | None = None,
) -> pd.Series:
    series: list[pd.Series] = []
    for df in df_list:
        _col = col or df.select_dtypes("number").columns[0]
        series.append(df[_col])
    return pd.concat(series, ignore_index=True)


def _infer_freq(df: pd.DataFrame) -> Optional[pd.Timedelta]:
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        freq_str = pd.infer_freq(df.index)
        if freq_str:
            return pd.to_timedelta(pd.tseries.frequencies.to_offset(freq_str))
        return None
    except Exception:
        return None


def _spectral_entropy(x: np.ndarray, *, eps: float = 1e-12) -> float:
    if x.size < 2:
        return np.nan          # 样本过少，无法计算

    _, pxx = periodogram(x, detrend="linear", scaling="density")

    # 将非有限值置零，避免后续出现 Inf / NaN
    pxx = np.where(np.isfinite(pxx), pxx, 0.0)

    total_power = np.sum(pxx)
    if not np.isfinite(total_power) or total_power <= eps:
        return 0.0             # 全 0（或无效）功率谱 -> 熵记为 0

    p_k = pxx / total_power    # 归一化功率谱
    return entropy(p_k, base=e)


def _distance_corr(x, y) -> float:
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    # 去掉 NaN / Inf
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask.ravel()], y[mask.ravel()]

    n = x.shape[0]
    if n < 2:
        return 0.0          # 样本不足，默认相关 0

    def _cent_dist(a):
        d = np.abs(a - a.T)
        d -= d.mean(axis=0)
        d -= d.mean(axis=1, keepdims=True)
        d += d.mean()
        return d

    A, B = _cent_dist(x), _cent_dist(y)
    dcov2 = (A * B).mean()
    dvar_x = (A * A).mean()
    dvar_y = (B * B).mean()

    denom = np.sqrt(dvar_x * dvar_y)
    return 0.0 if denom == 0 else np.sqrt(dcov2) / denom


def _iter_value_cols(df: pd.DataFrame, value_col: str | Sequence[str] | None):
    if value_col is None:
        return df.select_dtypes("number").columns
    if isinstance(value_col, str):
        return [value_col]
    return list(value_col)


def _vectorize_df(
    df: pd.DataFrame,
    *,
    value_cols: str | Sequence[str] | None = None,
    method: Literal["flatten", "summary"] = "summary", # {"flatten", "summary"}
    fillna: float | None = 0.0,
) -> np.ndarray:
    # 1. 选列
    if value_cols is None:
        num_df = df.select_dtypes("number")
        if num_df.empty:
            raise ValueError("当前 DataFrame 无数值列可用")
    else:
        # 转成 list 方便后续遍历
        if isinstance(value_cols, str):
            value_cols = [value_cols]
        missing = [c for c in value_cols if c not in df.columns]
        if missing:
            raise KeyError(f"DataFrame 缺少列：{missing}")
        num_df = df[list(value_cols)].select_dtypes("number")

    # 2. 缺失处理
    if fillna is None:
        num_df = num_df.dropna(axis=0, how="any")
    else:
        num_df = num_df.fillna(fillna)

    # 3. 向量化
    if method == "flatten":
        return num_df.to_numpy().ravel()

    feats = []
    for col in num_df.columns:
        x = num_df[col].to_numpy()
        feats.extend([
            x.mean(),
            x.std(ddof=1),
            *np.percentile(x, [0, 25, 50, 75, 100]),
            skew(x),
            kurtosis(x),
        ])
    return np.asarray(feats, dtype=float)


def _row_majority(values: np.ndarray) -> tuple[object or np.nan, bool]:
    vals = values[~pd.isna(values)]
    if vals.size == 0:
        return np.nan, False
    uniq, cnt = np.unique(vals, return_counts=True)
    max_cnt = cnt.max()
    modes = uniq[cnt == max_cnt]
    if modes.size == 1:
        return modes[0], False
    return modes, True


@numba.jit(nopython=True, cache=True)
def _create_dynamic_series_pca_numba(data: np.ndarray, window_size: int, step: int) -> np.ndarray:
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // step + 1
    dynamic_series = np.empty(n_windows, dtype=np.float64)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = data[start:end]

        # 手动实现 StandardScaler
        col_means = np.empty(n_features, dtype=np.float64)
        col_stds = np.empty(n_features, dtype=np.float64)
        for j in range(n_features):
            col_data = window[:, j]
            col_means[j] = np.mean(col_data)
            col_stds[j] = np.std(col_data)
            if col_stds[j] == 0:
                col_stds[j] = 1.0

        scaled_window = (window - col_means) / col_stds

        # 手动实现 PCA (n_components=1)
        if scaled_window.shape[1] > 0:
            cov_matrix = np.cov(scaled_window.T)
            if cov_matrix.ndim == 2:
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                sum_eigenvalues = np.sum(eigenvalues)
                if sum_eigenvalues > 1e-9:
                    explained_variance_ratio = np.max(eigenvalues) / sum_eigenvalues
                else:
                    explained_variance_ratio = 1.0
            else:
                explained_variance_ratio = 1.0
        else:
            explained_variance_ratio = 1.0

        dynamic_series[i] = explained_variance_ratio

    return dynamic_series


@numba.jit(nopython=True, cache=True)
def _create_dynamic_series_volatility_numba(data: np.ndarray, window_size: int, step: int) -> np.ndarray:
    n_samples, n_features = data.shape
    n_windows = (n_samples - window_size) // step + 1
    dynamic_series = np.empty(n_windows, dtype=np.float64)

    for i in range(n_windows):
        start = i * step
        end = start + window_size
        window = data[start:end]

        # 计算每个特征在窗口内的标准差
        stds = np.empty(n_features, dtype=np.float64)
        for j in range(n_features):
            stds[j] = np.std(window[:, j])

        # 计算所有特征标准差的平均值
        mean_std = np.mean(stds)
        dynamic_series[i] = mean_std

    return dynamic_series


def _create_dynamic_series(
        df: pd.DataFrame,
        window_size: int,
        step: int,
        method: str = 'pca'  # {'pca', 'volatility'}
) -> np.ndarray:
    df_numeric = df.select_dtypes(include=np.number)
    if df_numeric.empty:
        raise ValueError("The DataFrame contains no numeric columns to analyze.")
    if len(df_numeric) < window_size:
        raise ValueError(f"DataFrame length ({len(df_numeric)}) must be >= window size ({window_size}).")

    # 将DataFrame转换为Numba可以处理的NumPy数组
    data_numpy = df_numeric.to_numpy(dtype=np.float64)

    # 根据方法调用不同的Numba优化函数
    if method == 'pca':
        return _create_dynamic_series_pca_numba(data_numpy, window_size, step)
    elif method == 'volatility':
        return _create_dynamic_series_volatility_numba(data_numpy, window_size, step)
    else:
        raise ValueError("Method must be either 'pca' or 'volatility'")