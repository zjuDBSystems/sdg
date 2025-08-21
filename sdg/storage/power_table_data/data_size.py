from math import log, e
from typing import List, Sequence, Literal

import numpy as np
import pandas as pd
from pandas._libs import NaTType
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf

from sdg.storage.power_table_data import _infer_freq, _concat_series, _spectral_entropy, _iter_value_cols, _vectorize_df


def score_time_granularity(
        df_list: List[pd.DataFrame],
        expected: Sequence[pd.Timedelta | NaTType] = (
                pd.Timedelta("15min"),
                pd.Timedelta("30min"),
                pd.Timedelta("1h"),
                pd.Timedelta("2h"),
                pd.Timedelta("4h"),
        ),
        time_col: str = "datetime"
) -> float:
    if expected is None or len(expected) == 0:
        return 100.0
    found_freqs = set()
    for df in df_list:
        if df.index.empty:
            continue
        try:
            df_copy = df.copy()
            df_copy.index = pd.to_datetime(df_copy[time_col])
            freq = _infer_freq(df_copy)
            if freq:
                found_freqs.add(freq)
        except (ValueError, TypeError) as e:
            print(f"Warning: A DataFrame's index could not be converted to DatetimeIndex. Skipping. Error: {e}")
            continue
    found_and_expected = found_freqs.intersection(set(expected))
    score = len(found_and_expected) / len(expected) * 100
    return score


def score_seasonality_strength(
        df_list: List[pd.DataFrame],
        value_col: str | Sequence[str] | None = None,
        period: int = 96,
        method: str = 'stl', # {'stl', 'classical', 'acf'}
        robust: bool = False
) -> float:
    def _seasonality_strength_from_variance(resid: np.ndarray, seasonal: np.ndarray) -> float:
        resid = np.asarray(resid, dtype=float)
        seasonal = np.asarray(seasonal, dtype=float)
        mask = np.isfinite(resid) & np.isfinite(seasonal)
        resid, seasonal = resid[mask], seasonal[mask]

        if resid.size < 2:
            return 0.0

        num_var = np.var(resid, ddof=0)
        den_var = np.var(resid + seasonal, ddof=0)

        if den_var <= 0.0 or not np.isfinite(den_var):
            return 0.0

        s = 1 - num_var / den_var
        return max(0.0, float(s))

    scores: list[float] = []

    for df in df_list:

        for col in _iter_value_cols(df, value_col):
            ts = _concat_series([df], col)
            ts = ts.fillna(0)

            if len(ts) < 2 * period:
                continue

            try:
                s = 0.0
                if method == 'stl':
                    # STL
                    stl_result = STL(ts, period=period, robust=robust).fit()
                    s = _seasonality_strength_from_variance(stl_result.resid, stl_result.seasonal)

                elif method == 'classical':
                    # 经典分解
                    decomposition = seasonal_decompose(ts, period=period, model='additive')
                    s = _seasonality_strength_from_variance(decomposition.resid, decomposition.seasonal)

                elif method == 'acf':
                    # ACF
                    ts_diff = ts.diff().dropna() # 差分以消除趋势
                    if len(ts_diff) <= period: continue # 确保差分后仍足够长
                    acf_values = acf(ts_diff, nlags=period, fft=True)
                    s = max(0.0, float(acf_values[period]))

                else:
                    raise ValueError("Method must be one of 'stl', 'classical', or 'acf'")

                scores.append(s)

            except Exception as e:
                print(f"Could not process column '{col}' with method '{method}'. Error: {e}")
                continue

    return float(np.mean(scores) * 100) if scores else np.nan


def score_trend_strength(
        df_list: List[pd.DataFrame],
        value_col: str | Sequence[str] | None = None,
        period: int = 96,
        method: str = 'stl', # {'stl', 'classical', 'regression'}
        robust: bool = False
) -> float:

    def _trend_strength_from_variance(resid: np.ndarray, trend: np.ndarray) -> float:
        resid = np.asarray(resid, dtype=float)
        trend = np.asarray(trend, dtype=float)
        mask = np.isfinite(resid) & np.isfinite(trend)
        resid, trend = resid[mask], trend[mask]

        if resid.size < 2:
            return 0.0

        num_var = np.var(resid, ddof=0)
        den_var = np.var(resid + trend, ddof=0)

        if den_var <= 0.0 or not np.isfinite(den_var):
            return 0.0

        s = 1 - num_var / den_var
        return max(0.0, float(s))

    scores: list[float] = []

    for df in df_list:

        for col in _iter_value_cols(df, value_col):
            ts = _concat_series([df], col)
            ts = ts.fillna(0)


            if method in ['stl', 'classical'] and len(ts) < 2 * period:
                continue

            if method == 'regression' and len(ts) < 2:
                continue

            try:
                s = 0.0
                if method == 'stl':
                    stl_result = STL(ts, period=period, robust=robust).fit()
                    s = _trend_strength_from_variance(stl_result.resid, stl_result.trend)

                elif method == 'classical':
                    decomposition = seasonal_decompose(ts, period=period, model='additive')
                    s = _trend_strength_from_variance(decomposition.resid, decomposition.trend)

                elif method == 'regression':
                    # 准备回归数据 X(时间), y(值)
                    X = np.arange(len(ts)).reshape(-1, 1)
                    y = ts.values

                    # 拟合线性模型
                    model = LinearRegression()
                    model.fit(X, y)

                    # R-squared 本身就是一个 0-1 的强度度量
                    s = model.score(X, y)

                else:
                    raise ValueError("Method must be one of 'stl', 'classical', or 'regression'")

                scores.append(s)

            except Exception as e:
                print(f"Could not process column '{col}' with method '{method}'. Error: {e}")
                continue

    return float(np.mean(scores) * 100) if scores else np.nan


def score_primary_freq_strength(
        df_list: List[pd.DataFrame],
        value_col: str | Sequence[str] | None = None,
) -> float:
    scores: list[float] = []
    for df in df_list:

        for col in _iter_value_cols(df, value_col):
            x = _concat_series([df], col).fillna(0).to_numpy()
            H = _spectral_entropy(x)
            s = 1 - H / log(len(x) // 2)
            scores.append(s)
    return float(np.mean(scores) * 100) if scores else np.nan


def score_dataset_balance(
    df_list: List[pd.DataFrame],
    *,
    value_cols: str | Sequence[str] | None = None,
    n_clusters: int | None = None,
    vector_method: Literal["flatten", "summary"] = "flatten",
    scale: bool = True,
    random_state: int = 42,
) -> float:
    # 1. 向量化
    X = np.vstack([
        _vectorize_df(
            df.fillna(0),
            value_cols=value_cols,
            method=vector_method
        )
        for df in df_list
    ])

    # 2. 检查向量长度一致
    lens = {v.size for v in X}
    if len(lens) > 1:
        raise ValueError(f"Inconsistent vector lengths: {sorted(lens)}")

    # 3. 标准化
    if scale:
        X = StandardScaler().fit_transform(X)

    # 4. 聚类
    N = len(df_list)
    if n_clusters is None:
        n_clusters = max(2, round(np.sqrt(N)))

    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = kmeans.fit_predict(X)

    # 5. 熵计算
    counts = np.bincount(labels, minlength=n_clusters)
    p = counts / counts.sum()
    H = entropy(p, base=e)
    return float(H / log(n_clusters) * 100)
