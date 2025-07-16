'''Operators for domain knowledge introduct.
'''

from typing import override, Dict, List
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


class DomainKnowledgeOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file = kwargs.get('input_table_file', "shanxi_day_train_total.pkl")
        self.output_table_file = kwargs.get('output_table_file', "shanxi_day_train_total.pkl")
        self.domain_knowledge_csv_path = kwargs.get('output_table_file', "Plan_refine/延安发电1号机组.csv")

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
            name='DomainKnowledgeOperator',
            description='DomainKnowledgeOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "DomainKnowledgeOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        domain_knowledge_csv = pd.read_csv(os.path.join(dataset.dirs[0].data_path, self.domain_knowledge_csv_path))
        ls_df = self.batch_broadcast_daily_to_min(ls_df, domain_knowledge_csv, window_size=192)

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')


    def _broadcast_daily_onto_index(
        self,
        idx: pd.DatetimeIndex,
        daily_df: pd.DataFrame,
        *,
        prefix: str | None = None,
    ) -> pd.DataFrame:

        dates = idx.normalize()  # 截到 00:00:00

        # 缺失日期直接报错，防止静默填充
        missing = np.setdiff1d(dates.unique(), daily_df.index)
        if len(missing):
            raise KeyError(f"daily_df lacks date: {missing.astype(str).tolist()}")

        # 广播：.loc 支持重复索引返回等长结果
        day_expanded = daily_df.loc[dates].copy()
        day_expanded.index = idx                        # 对齐 15 min 时间戳
        if prefix:
            day_expanded.columns = [f"{prefix}{c}" for c in day_expanded.columns]
        return day_expanded


    def batch_broadcast_daily_to_min(
        self,
        intra_list: List[pd.DataFrame],
        daily_df: pd.DataFrame,
        *,
        window_size: int = 192,
        daily_prefix: str | None = "",
    ) -> List[pd.DataFrame]:

        if window_size % 96 != 0:
            raise ValueError("window_size must be a multiple of 96 (slots per day)")

        # 预处理日频数据
        daily_df = daily_df.copy()
        daily_df['datetime'] = pd.to_datetime(daily_df['datetime'], errors='coerce')
        daily_df.set_index('datetime', inplace=True)
        daily_df.fillna(0, inplace=True)

        out: List[pd.DataFrame] = []

        for i, intra_df in enumerate(intra_list, 1):
            if len(intra_df) != window_size:
                raise ValueError(f"Sample length {len(intra_df)} != window_size={window_size}")

            if 'datetime' not in intra_df.columns:
                raise KeyError(f"Sample #{i} lacks a 'datetime' column")

            intra_df = intra_df.copy()
            intra_df['datetime'] = pd.to_datetime(intra_df['datetime'], errors='coerce')

            if intra_df['datetime'].isna().any():
                bad_rows = intra_df[intra_df['datetime'].isna()].index.tolist()
                raise ValueError(f"Sample #{i} has unparsable datetime at rows: {bad_rows}")

            intra_df.set_index('datetime', inplace=True)

            daily_broadcast = self._broadcast_daily_onto_index(
                intra_df.index, daily_df, prefix=daily_prefix
            )

            combined = pd.concat([intra_df, daily_broadcast], axis=1)

            if combined.columns.duplicated().any():
                dup_cols = combined.columns[combined.columns.duplicated()].tolist()
                raise ValueError(f"Duplicated columns after concat: {dup_cols}")

            combined = (
                combined
                .reset_index()               # 把索引变回列
                .rename(columns={'index': 'datetime'})
            )

            combined.insert(0, 'datetime', combined.pop('datetime'))

            out.append(combined)

        return out




