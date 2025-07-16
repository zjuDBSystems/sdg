'''Operators for redundant feature removement.
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


class RedundantFeatureRemoveOperator(Operator):
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
            name='RedundantFeatureRemoveOperator',
            description='RedundantFeatureRemoveOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "RedundantFeatureRemoveOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        ls_df = self.remove_redundant_feature(ls_df, target_col=self.target_col)
        non_redundant_feature = ls_df[0].columns.tolist()

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    def remove_redundant_feature(
            self,
            arr_train: List[pd.DataFrame],
            target_col: str,
            corr_threshold: float = 0.8,
            importance_ratio: float = 0.1
    ) -> List[pd.DataFrame]:
        """
        参数:
            arr_train: 时序数据样本列表，每个元素是一个包含特征和目标列的DataFrame
            target_col: 目标列的列名
            corr_threshold: 相关性阈值，超过此值的特征被认为高度相关
            importance_ratio: 重要性比率阈值，低于此值的特征被认为可移除
        返回:
            移除冗余特征后的时序数据样本列表
        """
        if not arr_train:
            return []

        for df in arr_train:
            if target_col not in df.columns:
                raise ValueError(f"所有DataFrame必须包含目标列: {target_col}")

        combined_df = pd.concat(arr_train, ignore_index=True)

        feature_cols = [col for col in combined_df.columns if col != target_col]
        X = combined_df[feature_cols]
        y = combined_df[target_col]

        is_classification = y.dtype == 'object' or len(y.unique()) / len(y) < 0.05

        if is_classification:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            model = RandomForestClassifier(n_estimators=5, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=5, random_state=42)
            y_encoded = y

        model.fit(X, y_encoded)
        feature_importance = pd.Series(model.feature_importances_, index=feature_cols)

        corr_matrix = X.corr().abs()

        redundant_features = set()

        for i in range(len(feature_cols)):
            col1 = feature_cols[i]
            if col1 in redundant_features:
                continue

            for j in range(i + 1, len(feature_cols)):
                col2 = feature_cols[j]
                if col2 in redundant_features:
                    continue

                if corr_matrix.loc[col1, col2] > corr_threshold: # type: ignore
                    imp1 = feature_importance[col1]
                    imp2 = feature_importance[col2]

                    if abs(imp1 - imp2) / max(imp1, imp2) < importance_ratio:
                        redundant_features.add(col2)
                    else:
                        if imp1 < imp2:
                            redundant_features.add(col1)
                        else:
                            redundant_features.add(col2)

        out_list = []
        for df in arr_train:
            cols_to_keep = [col for col in df.columns if col not in redundant_features or col == target_col]
            out_list.append(df[cols_to_keep].copy())

        return out_list



