import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import pywt
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score

# 自相关系数
def acf_multiscale_all(csv_path: str,
                         time_col: str = "datetime",
                         lags: tuple = (1, 6, 24, 96),
                         norm: float = 0.2) -> pd.Series:
    df = (pd.read_csv(csv_path, parse_dates=[time_col])
            .select_dtypes(include=[np.number]))
    out = {}
    for col in df.columns:
        s = df[col].dropna().values
        # 常数或长度不足
        if len(s) < max(lags) + 1 or np.allclose(s, s[0]):
            out[col] = 0.0
            continue

        peaks = []
        for l in lags:
            if len(s) > l:
                peaks.append(acf(s, nlags=l, fft=True)[l])
        std_acf = np.std(peaks) if peaks else 0.0
        score0 = np.clip(std_acf / norm, 0, 1)
        out[col] = score0 * 100
    return pd.Series(out, name="acf_multiscale_score")


# 趋势强度
def trend_strength_all(csv_path,
                           time_col="datetime",
                           windows=(3, 6, 12, 24, 48),
                           gamma=0.8):
    # 1. 读入并预处理
    df = pd.read_csv(csv_path)
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)

    # 2. 自动选出所有数值型列
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scores = {}

    # 3. 对每一列分别计算趋势强度
    for col in num_cols:
        series = df[col].fillna(0).astype(float)

        # 总方差（避免除零）
        var_total = series.var(ddof=0) or 1e-9

        # 各尺度趋势能量占比
        ratios = []
        for w in windows:
            trend = series.rolling(window=w, min_periods=1, center=True).mean()
            ratios.append(trend.var(ddof=0) / var_total)

        mean_ratio = np.mean(ratios)
        score = min(mean_ratio / gamma, 1.0) * 100
        scores[col] = score

    return pd.Series(scores, name="acf_multiscale_score")


# 互信息
def cross_scale_mi_all(csv_path: str,
                         time_col: str = "datetime",
                         scales: tuple = (2, 4, 8),
                         q: int = 16) -> pd.Series:
    df = (pd.read_csv(csv_path, parse_dates=[time_col])
            .select_dtypes(include=[np.number]))
    out = {}
    for col in df.columns:
        s = df[col].dropna().values
        if len(s) == 0:
            out[col] = 0.0
            continue

        # 量化函数
        def quant(v):
            bins = np.linspace(v.min(), v.max(), q + 1)
            return np.digitize(v, bins) - 1

        base = quant(s)
        mi_list = []
        n = len(s)
        for m in scales:
            if n // m < 10:
                continue
            ds = s[::m]
            up = np.interp(np.arange(n), np.arange(0, n, m), ds)
            mi = mutual_info_score(base, quant(up))
            mi_list.append(mi)

        if mi_list:
            mi_avg = np.mean(mi_list)
            score = (mi_avg / np.log(q)) * 100
        else:
            score = 0.0
        out[col] = score

    return pd.Series(out, name="cross_scale_mi_score")
