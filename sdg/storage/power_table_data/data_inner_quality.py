from typing import List, Sequence, Optional

import pandas as pd


def score_missing_rate(
        df_list: List[pd.DataFrame],
) -> float:
    n_miss = sum(df.isna().sum().sum() for df in df_list)
    n_total = sum(df.size for df in df_list)
    return (1 - n_miss / n_total) * 100


def score_label_consistency(
        df_list: List[pd.DataFrame],
        *,
        label_col: str,
        vote_cols: Optional[Sequence[str]] = None,
        tie_policy: str = "skip",  # 可选值: 'skip' 或 'match_any'
        skip_nan: bool = True,
) -> float:
    import numpy as np
    total, ok = 0, 0

    for df in df_list:
        # 确保标签列存在，否则跳过此 DataFrame
        if label_col not in df.columns:
            continue

        # 确定用于投票的列
        if vote_cols is None:
            current_vote_cols = []
        else:
            current_vote_cols = list(vote_cols) if not isinstance(vote_cols, str) else [vote_cols]

        # 过滤掉数据中不存在的投票列
        current_vote_cols = [col for col in current_vote_cols if col in df.columns]
        if not current_vote_cols:
            continue

        # 先将投票列的值转换为整数部分
        def to_int_series(s):
            return s.apply(lambda v: int(v) if pd.notna(v) else pd.NA)
        int_votes = df[current_vote_cols].apply(to_int_series)

        # 手动实现逐行众数
        def row_mode(row):
            vals = [v for v in row if pd.notna(v)]
            if not vals:
                return [pd.NA]
            uniq, counts = np.unique(vals, return_counts=True)
            max_cnt = counts.max()
            mode_vals = uniq[counts == max_cnt]
            return list(mode_vals)
        mode_list = int_votes.apply(row_mode, axis=1)
        max_len = int(max(mode_list.map(len)))
        padded_modes = mode_list.apply(lambda x: x + [pd.NA] * (max_len - len(x)))
        modes = pd.DataFrame(padded_modes.tolist(), index=int_votes.index)

        if modes.empty:
            continue

        # 提取标签列，并同样将其转换为整数部分，以确保比较基准一致
        labels = df[label_col].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

        # 找出有有效投票结果的行
        valid_mask = modes.iloc[:, 0].notna()
        if not valid_mask.any():
            continue

        # 检测平局：当众数结果有超过一列时，第二列非空的位置即为平局
        if modes.shape[1] > 1:
            is_tie = modes.iloc[:, 1].notna()
        else:
            is_tie = pd.Series(False, index=modes.index)

        # 根据平局策略决定哪些行需要被计分
        if tie_policy == 'skip':
            # 只考虑有效且非平局的行
            rows_to_score_mask = valid_mask & ~is_tie
            if not rows_to_score_mask.any():
                continue

            total += np.count_nonzero(rows_to_score_mask)

            # 对于非平局情况，直接比较第一众数和标签
            ok += (modes.loc[rows_to_score_mask, 0] == labels[rows_to_score_mask]).sum()

        else:  # tie_policy == 'match_any'
            # 考虑所有有效的行
            rows_to_score_mask = valid_mask
            if not rows_to_score_mask.any():
                continue

            total += rows_to_score_mask.astype(int).sum()

            # 将所有有效的众数与对应的标签进行比较
            # .eq() 支持 DataFrame 和 Series 间的广播比较
            # .any(axis=1) 检查每一行是否有任意一个众数匹配标签
            matches = modes[rows_to_score_mask].eq(labels[rows_to_score_mask], axis=0).any(axis=1)
            ok += matches.sum()

    # 避免除以零错误
    return np.nan if total == 0 else ok / total * 100
