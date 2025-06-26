import numpy as np
import pandas as pd

def bucket_coverage_csv(csv_path, time_col="datetime",
                        freq="15 minute"):
    df = pd.read_csv(csv_path).fillna(0)

    df[time_col] = pd.to_datetime(df[time_col])
    ts_sorted = df[time_col].sort_values()

    step = pd.Timedelta(freq)
    full_range = pd.date_range(ts_sorted.iloc[0].floor("d"),
                               ts_sorted.iloc[-1].ceil("d") - step,
                               freq=step)

    coverage = len(ts_sorted) / len(full_range) if len(full_range) else 0
    return min(coverage, 1.0) * 100

def holiday_balance_csv(csv_path, time_col="datetime"):
    df = pd.read_csv(csv_path).fillna(0)
    df[time_col] = pd.to_datetime(df[time_col])

    p_workday = (df[time_col].dt.weekday < 5).mean()  # True=工作日
    return (1 - abs(p_workday - 0.5) / 0.5) * 100

def season_span_csv(csv_path, time_col="datetime"):
    df = pd.read_csv(csv_path).fillna(0)
    df[time_col] = pd.to_datetime(df[time_col])

    n_months = df[time_col].dt.month.nunique()
    return min(n_months / 12, 1.0) * 100