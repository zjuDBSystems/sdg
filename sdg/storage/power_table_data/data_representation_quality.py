import warnings
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from sdg.storage.power_table_data import _concat_series, _iter_value_cols


def score_stationarity_all(
        df_list: List[pd.DataFrame],
        value_col: str | Sequence[str] | None = None,
) -> float:
    def _stationarity_score(ts: np.ndarray) -> float:
        # 1. 清理和预检查
        ts = ts[np.isfinite(ts)]
        if ts.size < 10:
            return np.nan  # 样本过少，无法可靠判断，返回nan以便后续过滤

        # 2. 处理退化情况
        if np.all(ts == ts[0]):
            return 1.0

        # 使用线性回归R^2检查完美的线性序列
        try:
            # 注意：这里我们直接在numpy数组上操作，而不是pandas Series
            X = np.arange(len(ts)).reshape(-1, 1)
            y = ts
            model = LinearRegression()
            model.fit(X, y)
            r_squared = model.score(X, y)
            # 如果R²极度接近1，我们视其为完美的线性序列
            if r_squared > 0.9999:
                return 0.0  # 完美的线性序列是非平稳的，得分为0
        except Exception:
            pass

        # 3. 执行统计检验
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            try:
                p_adf = adfuller(ts, autolag="AIC")[1]
            except Exception:
                p_adf = 1.0

            try:
                p_kpss = kpss(ts, regression='c', nlags="auto")[1]
            except Exception:
                p_kpss = 0.0

        # 4. 合并为最终得分
        p_adf = np.clip(p_adf, 0.0, 1.0) if np.isfinite(p_adf) else 1.0
        p_kpss = np.clip(p_kpss, 0.0, 1.0) if np.isfinite(p_kpss) else 0.0

        return (1 - p_adf + p_kpss) / 2.0

    scores: list[float] = []
    for df in df_list:
        for col in _iter_value_cols(df, value_col):
            ts = _concat_series([df], col).fillna(0).to_numpy(dtype=float)
            scores.append(_stationarity_score(ts))

    valid_scores = [s for s in scores if not np.isnan(s)]

    return float(np.mean(valid_scores) * 100) if valid_scores else np.nan


def score_feature_readiness(
        df_list: List[pd.DataFrame],
        target_col: str,
        expected_features: List[str],
        date_col: str = "datetime",
        midpoint_tsnr: float = 2.0,
        coverage_weight: float = 0.5
) -> float:
    if not df_list: return 0.0
    try:
        df = pd.concat(df_list, ignore_index=True)
    except Exception:
        return 0.0

    # 如果需要计算质量，datetime和target列必须存在
    if coverage_weight < 1.0 and (target_col not in df.columns or date_col not in df.columns):
        # 如果无法计算质量分，但覆盖率权重不是100%，则无法给出有意义的分数
        # 或者可以只返回覆盖率分数，这里选择返回0
        return 0.0

    present_features = [f for f in expected_features if f in df.columns]
    coverage_score = (len(present_features) / len(expected_features)) * 100 if expected_features else 0.0

    # 如果只关心覆盖率，或者没有任何期望的特征存在，则直接返回覆盖率分数
    if coverage_weight == 1.0 or not present_features:
        return coverage_score

    quality_scores = []

    # 定义哪些是离散的，哪些是连续的
    discrete_features_def = ['month', 'day', 'weekday', 'hour', 'minute', 'holiday']
    continuous_features_def = ['elev', 'az', 'solar_elevation', 'solar_azimuth']  # 兼容不同命名

    # 计算离散特征的质量分
    for feature in [f for f in present_features if f in discrete_features_def]:
        grouped = df.groupby(feature)[target_col]
        if grouped.ngroups < 2: continue
        v_signal = grouped.mean().var(ddof=0)
        v_noise = grouped.var(ddof=0).mean()
        if v_noise > 1e-9:
            tsnr = v_signal / v_noise
            score = 100 * (tsnr / (midpoint_tsnr + tsnr))
            quality_scores.append(score)

    # 计算连续特征的质量分
    for feature in [f for f in present_features if f in continuous_features_def]:
        if df[feature].nunique() > 1:
            corr = df[[feature, target_col]].corr(numeric_only=True).iloc[0, 1]
            score = 100 * abs(corr)
            quality_scores.append(score)

    quality_score = np.mean(quality_scores) if quality_scores else 0.0

    final_score = (coverage_weight * coverage_score) + ((1 - coverage_weight) * quality_score)

    return final_score