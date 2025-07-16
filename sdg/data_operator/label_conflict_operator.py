'''Operators for label conflict processment.
'''

from typing import override, Dict, List, Sequence
import openai
import os
import pandas as pd
import tiktoken
import numpy as np
from astral import sun, LocationInfo
import chinese_calendar
from ..config import settings
from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType
import re
import pickle as pkl


class LabelConflictOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file = kwargs.get('input_table_file', "shanxi_day_train_total.pkl")
        self.output_table_file = kwargs.get('output_table_file', "shanxi_day_train_total.pkl")
        self.vote_cols = ['延安发电1号机组', '延安发电2号机组', '延热发电1号机组', '延热发电2号机组']
        self.target_col = "延安发电1号机组"

    @classmethod
    @override
    def accept(cls, data_type, task_type) -> bool:
        if data_type == DataType.TABLE and task_type == TaskType.AUGMENTATION:
            return True
        return False

    @classmethod
    @override
    def get_config(cls) -> list[Field]:
        return [
            Field('score_file', Field.FieldType.STRING, 'Score result file path', "./detailed_scores.csv")
        ]

    @classmethod
    @override
    def get_meta(cls) -> Meta:
        return Meta(
            name='LabelConflictOperator',
            description='LabelConflictOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "LabelConflictOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        ls_df = self.majority_vote_to_target(ls_df,
                                             vote_cols=self.vote_cols,
                                             target_col=self.target_col)

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')


    def majority_vote_to_target(
            self,
            arr_train: List[pd.DataFrame],
            vote_cols: Sequence[str],
            target_col: str,
            dropna: bool = True,
            tie_policy: str = "match_any"  # 可选值: {'skip', 'match_any'}
    ) -> List[pd.DataFrame]:
        out = []
        for df in arr_train:
            missing = set(vote_cols) - set(df.columns)
            if missing:
                raise KeyError(f"数据中缺少以下列: {missing}")

            new_df = df.copy()

            int_votes = new_df[list(vote_cols)].map(
                lambda v: int(v) if pd.notna(v) else pd.NA
            )

            # 对整数化后的投票列每一行计算众数（即多数票）
            def row_mode(row):
                vals = row.dropna().tolist()
                if len(vals) == 0:
                    return [pd.NA]
                cnts = pd.Series(vals).value_counts()
                max_cnt = cnts.max()
                mode_vals = list(cnts[cnts == max_cnt].index)
                return mode_vals
            
            mode_list = int_votes.apply(row_mode, axis=1)
            # 计算最大众数个数
            max_len = int(max(mode_list.map(len)))
            # 补齐每一行到最大长度
            padded_modes = mode_list.apply(lambda x: x + [pd.NA] * (max_len - len(x)))
            modes = pd.DataFrame(padded_modes.tolist(), index=int_votes.index)

            if modes.empty:
                # 如果没有有效数据，则用 NA 填充
                voted_values = pd.Series([pd.NA] * len(df), index=df.index, dtype='object')
            else:
                # 检查是否存在平局
                if modes.shape[1] > 1:
                    is_tie = modes.iloc[:, 1].notna()
                else:
                    is_tie = pd.Series(False, index=modes.index)

                # 根据平局策略进行处理
                if tie_policy == 'skip':
                    # 跳过平局，将其设置为 NA
                    voted_values = modes.iloc[:, 0].copy()
                    voted_values[is_tie] = pd.NA
                else:  # 'match_any'
                    # 出现平局时，使用第一个值
                    voted_values = modes.iloc[:, 0]

            # 直接将投票结果赋值给新的目标列
            new_df[target_col] = voted_values

            out.append(new_df)

        return out


