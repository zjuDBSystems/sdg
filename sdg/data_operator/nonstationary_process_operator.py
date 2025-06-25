'''Operators for non-stationary enhancement.
'''

from typing import override, Dict
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

class NonStationaryProcessOperator(Operator):
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', "timesfm")
        self.score_file = kwargs.get('score_file', "./detailed_scores.csv")


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
        cost["name"] = "MultiscaleEnhanceOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        df = pd.read_csv(dataset.dirs[0].data_path, na_values=['nan', 'None', ''])

        batch_size = 32
        context_len = 96
        horizon_len = 96
        total_len = context_len + horizon_len

        cols = [
            'datetime', '负荷预测', '风电总出力预测数值', '光伏总出力预测数值', '新能源总出力预测数值','非市场机组总出力预测',
            '外来电交易计划', '竞价空间', '延安发电1号机组', '延安发电1号机组运行状态'
        ]
        target_patterns = '延安发电1号机组'
        cov_num_patterns = [
            "负荷预测","风电总出力预测数值","光伏总出力预测数值","新能源总出力预测数值","非市场机组总出力预测","外来电交易计划","竞价空间",
        ]
        cov_cat_patterns = [
            "延安发电1号机组运行状态"
        ]

        df = df[cols]
        df.index = pd.to_datetime(df['datetime'])
        df = df.loc['2025-02':'2025-06']

        cols.remove('datetime')

        # 保存新数据
        df.to_csv(dataset.dirs[0].data_path[:-4]+'_done2.csv', index=False)
        print("对特定变量添加多尺度特征进行数据增强完成")





