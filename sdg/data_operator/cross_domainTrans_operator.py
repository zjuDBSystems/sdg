'''Operators for Cross Domain Transfer.
'''

from typing import override, Dict, List, Set
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
from sklearn.ensemble import RandomForestRegressor
import re
import pickle as pkl


class CrossDomainTransOperator(Operator):
    def __init__(self, **kwargs):
        self.input_table_file1 = kwargs.get('input_table_file', "shanxi_day_train_total_96_96.pkl")
        self.output_table_file1 = kwargs.get('output_table_file', "shanxi_day_train_total_96_96.pkl")
        self.input_table_file2 = kwargs.get('input_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.output_table_file2 = kwargs.get('output_table_file2', "shanxi_day_train_total_192_192.pkl")
        self.input_table_file3 = kwargs.get('input_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.output_table_file3 = kwargs.get('output_table_file3', "shanxi_day_train_total_384_384.pkl")
        self.cross_domain_csv_path = kwargs.get('cross_domain_csv', "ShandongEP/ShandongEP_2022.csv")
        self.n_estimators = 64

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
            name='CrossDomainTransOperator',
            description='CrossDomainTransOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "CrossDomainTransOperator"
        return cost


    @override
    def execute(self, dataset):

        cross_domain_csv = pd.read_csv(os.path.join(dataset.dirs[0].data_path, self.cross_domain_csv_path))
        # file_96_96
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file1), "rb"))
        ls_df = self.augment_reserve_samples(ls_df,
                                             cross_domain_csv,
                                             n_estimators=self.n_estimators)
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file1), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        # file_192_192
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file2), "rb"))
        ls_df = self.augment_reserve_samples(ls_df,
                                             cross_domain_csv,
                                             n_estimators=self.n_estimators)

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file2), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        # file_384_384
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file3), "rb"))
        ls_df = self.augment_reserve_samples(ls_df,
                                             cross_domain_csv,
                                             n_estimators=self.n_estimators)
        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file3), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    def augment_reserve_samples(
            self,
            target_dfs: List[pd.DataFrame],
            source_df: pd.DataFrame,
            *,
            pos_col: str = "正备用预测",
            neg_col: str = "负备用预测",
            n_estimators: int = 400,
            random_state: int = 42,
    ) -> List[pd.DataFrame]:
        # 1. 选出初始的公共列
        initial_common_cols: Set[str] = (
            set(source_df.columns)
            .intersection(target_dfs[0].columns)
            .difference({pos_col, neg_col})
        )

        if not initial_common_cols:
            raise ValueError("There are no shared feature columns between the source domain and the target domain.")

        # 这确保了日期时间字符串或其他对象类型被排除。
        numeric_cols = source_df.select_dtypes(include=np.number).columns
        common_cols = sorted(list(initial_common_cols.intersection(numeric_cols)))

        # 筛选后再次检查
        if not common_cols:
            raise ValueError("There are no shared numerical type feature columns between the source domain and the target domain.")

        # 2. 拆分特征/标签
        X_src = source_df[common_cols]
        y_src = source_df[[pos_col, neg_col]]

        # 3. 建立管道并训练
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            n_jobs=-1,
            oob_score=False,
            random_state=random_state,
        )
        model.fit(X_src, y_src)

        # 4. 批量推断增广
        aug_list: List[pd.DataFrame] = []
        for df in target_dfs:
            # 确保目标域也只使用这些数值列进行预测
            X_tgt = df[common_cols]
            preds = model.predict(X_tgt)

            out = df.copy()
            out[pos_col] = preds[:, 0]
            out[neg_col] = preds[:, 1]
            aug_list.append(out)

        return aug_list



