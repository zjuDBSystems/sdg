'''Operators for time-feature enhancement.
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


class TimeFeatureEnhanceOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file1 = kwargs.get('input_table_file', "shanxi_day_train_total_96_96.pkl")
        self.output_table_file1 = kwargs.get('output_table_file', "shanxi_day_train_total_96_96.pkl")
        self.input_table_file2 = kwargs.get('input_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.output_table_file2 = kwargs.get('output_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.input_table_file3 = kwargs.get('input_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.output_table_file3 = kwargs.get('output_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.loc = LocationInfo("Yanan", "CN", "Asia/Shanghai", 36.5853932, 109.4828549).observer
        self.time_enhancement = kwargs.get('time_enhancement', True)

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
            name='TimeFeatureEnhanceOperator',
            description='TimeFeatureEnhanceOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "TimeFeatureEnhanceOperator"
        return cost


    @override
    def execute(self, dataset):
        # file_96_96
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file1), "rb"))
        ls_df = self.time_enhance(ls_df, loc=self.loc, enhancement=self.time_enhancement)
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file1), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        # file_192_192
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file2), "rb"))
        ls_df = self.time_enhance(ls_df, loc=self.loc, enhancement=self.time_enhancement)
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file2), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        # file_384_384
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file3), "rb"))
        ls_df = self.time_enhance(ls_df, loc=self.loc, enhancement=self.time_enhancement)
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file3), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')


    def time_enhance(self, arr_train: List[pd.DataFrame], loc, enhancement: bool):
        if enhancement:
            for df in arr_train:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                dt = df['datetime'].dt

                df['month'] = dt.month
                df['day'] = dt.day
                df['weekday'] = dt.weekday
                df['hour'] = dt.hour
                df['minute'] = dt.minute

                df['holiday'] = df['datetime'].map(lambda t: int(chinese_calendar.is_holiday(t)))

                df["elev"] = [sun.elevation(loc, t) for t in df.datetime]
                df["az"] = [sun.azimuth(loc, t) for t in df.datetime]

                if 'datetime' in df.columns:
                    df.set_index('datetime', inplace=True)
        else:
            for df in arr_train:
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df.set_index('datetime', inplace=True)
        return arr_train

