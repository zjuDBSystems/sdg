'''Operators for non-stationary enhancement.
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


class NonStationaryProcessOperator(Operator):
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
            name='NonStationaryProcessOperator',
            description='Non-Stationary Process.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "NonStationaryProcessOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')





