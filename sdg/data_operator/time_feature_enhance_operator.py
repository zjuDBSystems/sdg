'''Operators for time enhancement.
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


class TimeFeatureEnhanceOperator(Operator):
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
            name='TimeFeatureEnhanceOperator',
            description='Time feature enhancement.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "TimeFeatureEnhanceOperator"
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
            "elev","az"
        ]
        cov_cat_patterns = [
            "延安发电1号机组运行状态","month","day","weekday","hour","minute","holiday",
        ]

        df = df[cols]
        df.index = pd.to_datetime(df['datetime'])
        df = df.loc['2025-02':'2025-06']

        cols_tmp = ['负荷预测', '风电总出力预测数值', '光伏总出力预测数值', '新能源总出力预测数值',
                    '非市场机组总出力预测', '外来电交易计划', '竞价空间', target_patterns]

        loc = LocationInfo("Yanan", "CN", "Asia/Shanghai", 36.5853932, 109.4828549).observer

        df['datetime'] = pd.to_datetime(df['datetime'])
        df['month'] = df.datetime.apply(lambda row: row.month).astype(int)
        df['day'] = df.datetime.apply(lambda row: row.day).astype(int)
        df['weekday'] = df.datetime.apply(lambda row: row.weekday()).astype(int)
        df['hour'] = df.datetime.apply(lambda row: row.hour).astype(int)
        df['minute'] = df.datetime.apply(lambda row: row.minute).astype(int)
        df["holiday"] = df.datetime.apply(lambda row: int(chinese_calendar.is_holiday(row))).astype(int)

        df["elev"] = [sun.elevation(loc, t) for t in df.datetime]  # 高度角
        df["az"] = [sun.azimuth(loc, t) for t in df.datetime]  # 方位角

        # 保存新数据
        df.to_csv(dataset.dirs[0].data_path[:-4]+'_done3.csv', index=False)
        print("提取时间特征作为协变量进行数据增强")




