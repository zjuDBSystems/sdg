import numpy as np
import pandas as pd
from numpy.fft import rfft
from scipy.stats import entropy

# 主频比例
def energy_ratio_all(csv_path: str, time_col: str = "datetime",
                     k: int = 10, gamma: float = 0.6) -> pd.Series:
    df = (pd.read_csv(csv_path, parse_dates=[time_col])
            .select_dtypes(include=[np.number]))
    out = {}
    for col in df.columns:
        s = df[col].values
        # 1. 计算功率谱
        power = np.abs(rfft(s - s.mean()))**2
        total = power.sum()
        if total <= 0:
            # 全零序列或能量为 0
            score = 0.0
        else:
            # 2. 取能量最大的 k 个分量之和
            if k > len(power):
                k = len(power)
            topk = np.partition(power, -k)[-k:]
            topk_energy = topk.sum()
            # 3. 归一化比例
            ratio = topk_energy / total / gamma
            # 4. 截断并转百分制
            score = min(ratio, 1.0) * 100
        out[col] = score
    return pd.Series(out, name="energy_ratio")

# 谱熵
def spectral_entropy_all(csv_path: str, time_col: str = "datetime") -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=[time_col]).select_dtypes(include=[np.number])
    out = {}
    for col in df.columns:
        s = df[col].values
        power = np.abs(rfft(s - s.mean()))**2
        total = power.sum()
        if total <= 0:
            score = 0.0
        else:
            p = power / total
            # 用 scipy.stats.entropy，保证 p 没有 nan
            score = (1 - entropy(p) / np.log(len(p))) * 100
        out[col] = score
    return pd.Series(out, name="spectral_entropy")