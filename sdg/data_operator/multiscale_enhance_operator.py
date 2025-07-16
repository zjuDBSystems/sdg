'''Operators for multiscale enhancement.
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


class MultiDownsampleOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file1 = kwargs.get('input_table_file1', "shanxi_day_train_total_96_96.pkl")
        self.output_table_file1 = kwargs.get('output_table_file1', "shanxi_day_train_total_96_96.pkl")
        self.input_table_file2 = kwargs.get('input_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.output_table_file2 = kwargs.get('output_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.input_table_file3 = kwargs.get('input_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.output_table_file3 = kwargs.get('output_table_file3', "shanxi_day_train_total_384_384.pkl")


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
            name='MultiDownsampleOperator',
            description='MultiDownsampleOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "MultiDownsampleOperator"
        return cost


    @override
    def execute(self, dataset):
        ls_df = []
       # file_96_96
        ls_df_96_96 = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file1), "rb"))
        ls_df_96_96 = self.downsample(ls_df_96_96, level=1)
        ls_df.extend(ls_df_96_96)
        # file_192_192
        ls_df_192_192 = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file2), "rb"))
        ls_df_192_192 = self.downsample(ls_df_192_192, level=2)
        ls_df.extend(ls_df_192_192)
        # file_384_384
        ls_df_384_384 = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file3), "rb"))
        ls_df_384_384 = self.downsample(ls_df_384_384, level=4)
        ls_df.extend(ls_df_384_384)

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file1), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    def pyramid_downsample(
            self,
            df: pd.DataFrame,
            level: int = 1,
    ) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("The DataFrame's index must be a DatetimeIndex for this operation.")

        if level <= 1:
            return df.copy()

        downsampled_data = df.groupby(np.arange(len(df)) // level).mean()
        new_index = df.index[::level]
        new_index = new_index[:len(downsampled_data)]
        downsampled_data.index = new_index
        return downsampled_data


    def downsample(self, arr_train: List[pd.DataFrame], level: int) -> List[pd.DataFrame]:
        return [self.pyramid_downsample(df, level=level) for df in arr_train]








