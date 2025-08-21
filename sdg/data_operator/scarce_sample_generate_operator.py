'''Operators for scarce sample generation.
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


class ScarceSampleGenerateOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file = kwargs.get('input_table_file', "shanxi_day_train_total_96_96.pkl")
        self.output_table_file = kwargs.get('output_table_file', "shanxi_day_train_total_96_96.pkl")
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
            name='ScarceSampleGenerateOperator',
            description='ScarceSampleGenerateOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "ScarceSampleGenerateOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        samples_idx = self.extreme_value_detector(ls_df, target_col=self.target_col)
        ls_df += self.generate_extreme_value_samples(ls_df, target_col=self.target_col, samples_index=samples_idx,level=[2, 4, 8])

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')


    def extreme_value_detector(
            self,
            arr_train: List[pd.DataFrame],
            target_col: str
    ) -> List[int]:
        threshold = 1.5
        idx_list = []

        for i, df in enumerate(arr_train):
            # 确保目标列是数值类型
            if not pd.api.types.is_numeric_dtype(df[target_col]):
                continue  # 跳过非数值列
            
            target_series = df[target_col].dropna()  # 移除缺失值
            if len(target_series) == 0:
                continue  # 跳过空列
            
            q1 = np.quantile(target_series, 0.25)
            q3 = np.quantile(target_series, 0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            if np.count_nonzero(np.logical_or(target_series < lower_bound, target_series > upper_bound)) >= 1:
                idx_list.append(i)

        return idx_list

    def generate_extreme_value_samples(
            self,
            arr_train: List[pd.DataFrame],
            target_col: str,
            samples_index: List[int],
            level: List[int]
    ) -> List[pd.DataFrame]:
        out_list: List[pd.DataFrame] = []

        for idx in samples_index:
            original_df = arr_train[idx]
            generated_df = original_df.copy(deep=True)
            target_data = generated_df[target_col].copy()
            
            # 只处理数值类型的特征列
            feature_cols = [col for col in generated_df.columns if col != target_col 
                           and pd.api.types.is_numeric_dtype(generated_df[col])]

            for col in feature_cols:
                # 保存原始数据长度用于后续检查
                original_length = len(generated_df)
                # 确保数据是数值类型且没有缺失值
                original_data = generated_df[col].replace([np.inf, -np.inf], np.nan).values
                
                # 处理每个窗口大小
                processed_data = original_data.copy()  # 默认使用原始数据
                for win_size in level:
                    # 确保窗口大小有效
                    if win_size < 2 or win_size > len(original_data):
                        continue
                        
                    # 计算移动平均，确保结果长度与原始数据一致
                    cumsum = np.cumsum(np.insert(original_data, 0, 0))
                    moving_avg = (cumsum[win_size:] - cumsum[:-win_size]) / win_size
                    
                    # 确保移动平均结果可以正确插值到原始长度
                    if len(moving_avg) < original_length:
                        # 使用线性插值确保长度匹配
                        x_old = np.linspace(0, original_length - 1, len(moving_avg))
                        x_new = np.arange(original_length)
                        processed_data = np.interp(x_new, x_old, moving_avg)
                    
                    # 验证长度是否匹配
                    if len(processed_data) != original_length:
                        # 如果不匹配，使用原始数据
                        processed_data = original_data.copy()
                        print(f"警告: 列 {col} 处理后长度不匹配，使用原始数据")
                
                # 确保赋值的数据长度与DataFrame索引长度一致
                generated_df[col] = processed_data

            generated_df[target_col] = target_data
            out_list.append(generated_df)

        return out_list

