import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf

# ADF
def adf_nonstationarity_all(csv_path: str, time_col: str = "datetime") -> pd.Series:
    """
    对每个数值列做 ADF 单位根检验，
    返回 p-value * 100，p-value 越大（越非平稳）分数越高；平稳或常数序列得分 0。
    """
    df = (pd.read_csv(csv_path, parse_dates=[time_col])
            .select_dtypes(include=[np.number]))
    out = {}
    for col in df.columns:
        s = df[col].dropna().values

        # 常数序列或太短时，视作最平稳，得 0 分
        if len(s) < 3 or np.all(s == s[0]):
            out[col] = 0.0
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            try:
                pvalue = adfuller(s, autolag="AIC")[1]
                score = np.clip(pvalue * 100, 0, 100)
                out[col] = float(score)
            except ValueError:
                # 其它异常也当作完全平稳
                out[col] = 0.0

    return pd.Series(out, name="adf_nonstationarity")


# 滚动方差
def rolling_var_cv_all(csv_path: str, time_col: str = "datetime",
                       window: int = 7 * 24) -> pd.Series:
    """
    对每个数值列计算滚动方差序列的 CV（std/mean），
    CV 越大（方差波动越剧烈）说明越非平稳，分数越高；平稳序列得分接近 0。
    """
    df = (pd.read_csv(csv_path, parse_dates=[time_col])
            .select_dtypes(include=[np.number]))
    out = {}
    for col in df.columns:
        s = df[col]
        var_series = s.rolling(window, min_periods=window // 2).var()
        cv = var_series.std() / (var_series.mean() + 1e-9)
        out[col] = np.clip(cv, 0, 1) * 100
    return pd.Series(out, name="rolling_var_cv")


# 赫斯特指数
def hurst_all(csv_path: str, time_col: str = "datetime") -> pd.Series:
    """
    对每个数值列估计 Hurst 指数 H，
    |H-0.5|*2*100 越高说明越非平稳（强趋势或反持久性），分数越高；H≈0.5（随机游走）得分越低。
    """
    df = (pd.read_csv(csv_path, parse_dates=[time_col])
            .select_dtypes(include=[np.number]))
    out = {}
    eps = 1e-9
    for col in df.columns:
        s = pd.Series(df[col].dropna().values)
        n = len(s)
        lags = np.arange(2, min(n // 2, 100))
        tau = [np.log10(s.diff(lag).dropna().std() + eps) for lag in lags]
        if len(lags) < 2 or np.allclose(tau, tau[0]):
            H = 0.5
        else:
            H = np.polyfit(np.log10(lags), tau, 1)[0]
        out[col] = min(abs(H - 0.5) * 2, 1) * 100
    return pd.Series(out, name="hurst")
