'''Operators for main frequency enhancement.
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


class MainFrequencyEnhanceOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file = kwargs.get('input_table_file', "shanxi_day_train_total.pkl")
        self.output_table_file = kwargs.get('output_table_file', "shanxi_day_train_total.pkl")
        self.topK = 10

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
            name='MainFrequencyEnhanceOperator',
            description='MainFrequencyEnhanceOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "MainFrequencyEnhanceOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        ls_df += self.fft_extract(ls_df, self.topK)


        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    
    def fft_extract_target(
        self,
        arr_train: List[pd.DataFrame],
        target_col: str,
        topK: int = 5
    ) -> List[pd.DataFrame]:
        out_list: List[pd.DataFrame] = []

        for df in arr_train:
            if target_col not in df.columns:
                raise KeyError(f"column {target_col} not exists in DataFrame")

            filt_df = df.copy(deep=True)
            x = df[target_col].to_numpy()
            N = x.size

            fft_vals = np.fft.fft(x)
            amps = np.abs(fft_vals)

            top_idx = np.argsort(amps)[::-1][1 : topK + 1]

            mask = np.zeros(N, dtype=bool)
            mask[top_idx] = True
            mask[-top_idx % N] = True

            fft_filtered = np.zeros_like(fft_vals, dtype=complex)
            fft_filtered[mask] = fft_vals[mask]

            filt_df[target_col] = np.fft.ifft(fft_filtered).real

            out_list.append(filt_df)

        return out_list



    def fft_extract(
        self,
        arr_train: List[pd.DataFrame],
        topK: int = 5
    ) -> List[pd.DataFrame]:

        out_list: List[pd.DataFrame] = []

        for df in arr_train:
            filt_df = df.copy(deep=True)

            for col in df.columns:
                x = df[col].to_numpy()
                N = x.size

                fft_vals = np.fft.fft(x)
                amps = np.abs(fft_vals)

                top_idx = np.argsort(amps)[::-1][1 : topK + 1]

                mask = np.zeros(N, dtype=bool)
                mask[top_idx] = True
                mask[-top_idx % N] = True

                fft_filtered = np.zeros_like(fft_vals, dtype=complex)
                fft_filtered[mask] = fft_vals[mask]

                filt_df[col] = np.fft.ifft(fft_filtered).real

            out_list.append(filt_df)

        return out_list






