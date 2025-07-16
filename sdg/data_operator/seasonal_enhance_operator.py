'''Operators for seasonal enhancement.
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
from statsmodels.tsa.seasonal import STL


class SeasonalEnhanceOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file = kwargs.get('input_table_file', "shanxi_day_train_total_96_96.pkl")
        self.output_table_file = kwargs.get('output_table_file', "shanxi_day_train_total_96_96.pkl")


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
            name='SeasonalEnhanceOperator',
            description='SeasonalEnhanceOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "SeasonalEnhanceOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        ls_df += self.seasonal_extract(ls_df)

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    def seasonal_extract(
        self,
        arr_train: List[pd.DataFrame],
        *,
        method: str = "ma",
        window: int = 25,
        center: bool = True,
        period: int | None = 96,
        robust: bool = True,
    ) -> List[pd.DataFrame]:

        if method not in {"ma", "stl"}:
            raise ValueError("method must be either 'ma' or 'stl'")

        seasonal_list: List[pd.DataFrame] = []

        for df in arr_train:
            if method == "ma":
                trend_df = (
                    df.rolling(window=window, center=center, min_periods=1)
                    .mean()
                    .bfill()
                    .ffill()
                )
                seasonal_df = (df - trend_df).bfill().ffill()

            else:
                if period is None:
                    raise ValueError("'period' must be provided when method='stl'")

                seasonal_cols: list[pd.Series] = []
                for col in df.columns:
                    stl_res = STL(df[col], period=period, robust=robust).fit()
                    seasonal_cols.append(stl_res.seasonal.rename(col))
                seasonal_df = pd.concat(seasonal_cols, axis=1)

            seasonal_list.append(seasonal_df)

        return seasonal_list





