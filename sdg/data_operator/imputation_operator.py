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


class NonStationaryProcessOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file = kwargs.get('input_table_file', "shanxi_day_train_total.pkl")
        self.output_table_file = kwargs.get('output_table_file', "shanxi_day_train_total.pkl")
        self.imputation_model_path = kwargs.get('imputation_model_path', "imputation_model_checkpoints/checkpoints_96_96.pypots")

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

        ls_df = self.saits_impute(ls_df, 
                                  model_path=os.path.join(dataset.dirs[0].data_path,self.imputation_model_path))

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
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
            epochs=100, patience=5, device="cuda" if torch.cuda.is_available() else "cpu"
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
        return [pd.DataFrame(imputed[i], columns=cols, index=index_t) for i in range(len(arr_train))]



