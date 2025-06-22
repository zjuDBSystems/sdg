import numpy as np
import pandas as pd

def bucket_coverage_csv(csv_path, time_col="datetime",
                        freq="1h"):
    """
    计算离散时间覆盖率:
        覆盖率 = 实际时间戳数量 / 理论完整栅格数量
    - freq: pandas 频率字符串，如 '1h'、'15min'
    """
    # ① 读全表 → 缺失补 0
    df = pd.read_csv(csv_path).fillna(0)

    # ② 时间列转 datetime 并排序
    df[time_col] = pd.to_datetime(df[time_col])
    ts_sorted = df[time_col].sort_values()

    # ③ 构造理论完整时间栅格
    step = pd.Timedelta(freq)
    full_range = pd.date_range(ts_sorted.iloc[0].floor("d"),
                               ts_sorted.iloc[-1].ceil("d") - step,
                               freq=step)

    # ④ 计算覆盖率并映射为得分
    coverage = len(ts_sorted) / len(full_range) if len(full_range) else 0
    return min(coverage, 1.0) * 100

def holiday_balance_csv(csv_path, time_col="datetime"):
    """
    工作日样本占比 p，目标期望 0.5：
        得分 = (1 - |p-0.5| / 0.5) * 100
    """
    df = pd.read_csv(csv_path).fillna(0)
    df[time_col] = pd.to_datetime(df[time_col])

    p_workday = (df[time_col].dt.weekday < 5).mean()  # True=工作日
    return (1 - abs(p_workday - 0.5) / 0.5) * 100

def season_span_csv(csv_path, time_col="datetime"):
    """
    不同月份出现个数 / 12 → 0~1，再 ×100
    """
    df = pd.read_csv(csv_path).fillna(0)
    df[time_col] = pd.to_datetime(df[time_col])

    n_months = df[time_col].dt.month.nunique()
    return min(n_months / 12, 1.0) * 100

def solar_angle_coverage_csv(csv_path,
                             time_col="datetime",
                             lat_deg=35.0) -> float:
    """
    计算太阳辐射角覆盖度，不再依赖 value_col。
    只读取时间列，并对缺失时间前向填充。
    """
    # 1. 只读时间列
    df = pd.read_csv(csv_path, usecols=[time_col])
    # 2. 转 datetime 并前向填充缺失
    t = pd.to_datetime(df[time_col]).fillna(0)

    # 3. 计算简化太阳高度角（忽略方程时差）
    phi = np.radians(lat_deg)
    day_of_year = t.dt.dayofyear.values
    hour_decimal = t.dt.hour + t.dt.minute / 60 + t.dt.second / 3600
    decl = np.radians(23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365)))
    H = np.radians(15 * (hour_decimal - 12))
    alt = np.arcsin(
        np.sin(phi) * np.sin(decl) +
        np.cos(phi) * np.cos(decl) * np.cos(H)
    )
    alt_deg = np.degrees(alt)

    # 4. 分档统计：[-90,0)、[0,30)、[30,60)、[60,90]
    bins = [-91, 0, 30, 60, 91]
    present = np.unique(np.digitize(alt_deg, bins))
    k = len(present)  # 最多 4 档

    # 5. 归一到 [0,100]
    return (k / 4) * 100