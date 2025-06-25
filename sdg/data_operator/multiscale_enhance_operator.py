'''Operators for multi-scale enhancement.
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

class MultiscaleEnhanceOperator(Operator):
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
            name='MultiscaleEnhanceOperator',
            description='Multi-scale enhancement.'
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

        levels = (2, 4)
        interp = 'linear'
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
        df = pd.concat([df, self.pyramid_downsample_concat(df[cols], levels, interp)], axis=1)
        target_cols = [c for c in df.columns if re.search(target_patterns, c)]
        target_cols.remove(target_patterns)
        df[target_cols] = df[target_cols].shift(horizon_len)
        df = df.iloc[horizon_len:].reset_index(drop=True)

        # 保存新数据
        df.to_csv(dataset.dirs[0].data_path[:-4]+'_done4.csv', index=False)
        print("对特定变量添加多尺度特征进行数据增强完成")


    def pyramid_downsample_concat(
            self,
            df: pd.DataFrame,
            levels=(1, 2, 4),
            interp: str = "linear",
            prefix: str = "pdc",
    ) -> pd.DataFrame:
        """
        多尺度均值池化后插值回原长度并拼接列

        Parameters
        ----------
        df : DataFrame
            原始序列，Index 已按时间或整数排序
        levels : tuple[int]
            下采样倍率 r；r=1 保留原序列
        interp : str
            reindex 后的插值方法，见 `pd.Series.interpolate`
        prefix : str
            给新列加的前缀；会得到形如 f"{prefix}{r}_{col}" 的列名

        Returns
        -------
        DataFrame
            列数 = 原列数 × len(levels)，索引与 df 完全一致
        """
        n = df.shape[0]
        out_list = []

        for r in levels:
            if r == 1:
                out = df.copy()
            else:
                groups = np.arange(n) // r
                down = df.groupby(groups).mean()

                new_index = np.linspace(0, len(down) - 1, n)
                up = (
                    down.reindex(new_index)
                    .interpolate(method=interp)
                    .set_index(df.index)
                )
                out = up

            out = out.add_prefix(f"{prefix}{r}_")
            out_list.append(out)

        mixed = pd.concat(out_list, axis=1)
        return mixed



