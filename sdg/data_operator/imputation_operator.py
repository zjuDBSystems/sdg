'''Operators for imputation.
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
import torch
from pypots.imputation import SAITS
from pathlib import Path
from statsmodels.tsa.seasonal import STL


class ImputationOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file1 = kwargs.get('input_table_file', "shanxi_day_train_total_96_96.pkl")
        self.output_table_file1 = kwargs.get('output_table_file', "shanxi_day_train_total_96_96.pkl")
        self.input_table_file2 = kwargs.get('input_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.output_table_file2 = kwargs.get('output_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.input_table_file3 = kwargs.get('input_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.output_table_file3 = kwargs.get('output_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.imputation_model_path1 = kwargs.get('imputation_model_path1', "imputation_model_checkpoints/checkpoints_96_96.pypots")
        self.imputation_model_path2 = kwargs.get('imputation_model_path2', "imputation_model_checkpoints/checkpoints_192_192.pypots")
        self.imputation_model_path3 = kwargs.get('imputation_model_path3', "imputation_model_checkpoints/checkpoints_384_384.pypots")

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
            name='ImputationOperator',
            description='ImputationOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "ImputationOperator"
        return cost


    @override
    def execute(self, dataset):
        # file_96_96
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file1), "rb"))
        ls_df = self.saits_impute(ls_df,model_path=os.path.join(dataset.dirs[0].data_path, self.imputation_model_path1))
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file1), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        # file_192_192
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file2), "rb"))
        ls_df = self.saits_impute(ls_df,model_path=os.path.join(dataset.dirs[0].data_path, self.imputation_model_path2))
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file2), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        # file_384_384
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file3), "rb"))
        ls_df = self.saits_impute(ls_df,model_path=os.path.join(dataset.dirs[0].data_path, self.imputation_model_path3))
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file3), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    def saits_impute(self, arr_train, model_path=None):
        X = np.stack([df.values for df in arr_train]).astype(np.float32)

        model = SAITS(
            n_steps=X.shape[1],
            n_features=X.shape[2],
            d_model=256, n_heads=4, d_ffn=512,
            n_layers=3, dropout=0.2,
            d_k=64, d_v=64,
            epochs=10, patience=3, device="cuda" if torch.cuda.is_available() else "cpu"
        )

        if model_path is not None and Path(model_path).exists():
            model.load(model_path)
        else:
            model.fit({"X": X})
            if model_path is not None:
                model.save(model_path)

        with torch.no_grad():
            imputed = model.impute({"X": X})

        cols = arr_train[0].columns
        index_t = arr_train[0].index
        return [pd.DataFrame(imputed[i], columns=cols, index=index_t) for i, _ in enumerate(arr_train)]



