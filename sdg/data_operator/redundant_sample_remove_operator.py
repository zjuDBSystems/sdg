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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class RedundantSampleRemoveOperator(Operator):
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
            name='RedundantSampleRemoveOperator',
            description='RedundantSampleRemoveOperator.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "RedundantSampleRemoveOperator"
        return cost


    @override
    def execute(self, dataset):
        # files
        ls_df = pkl.load(open(os.path.join(dataset.dirs[0].data_path, self.input_table_file), "rb"))

        ls_df = self.remove_redundant_sample(ls_df, target_col=self.target_col)

        with open(os.path.join(dataset.dirs[0].data_path, self.output_table_file), "wb") as file:
            pkl.dump(ls_df, file, protocol=5)
        
        print(f'{self.get_meta().name}算子执行完成')

    def remove_redundant_sample(
            self,
            arr_train: List[pd.DataFrame],
            target_col: str,
            n_clusters: int = 5,
            random_state: int | None = None
    ) -> List[pd.DataFrame]:
        if random_state is None:
            random_state = np.random.RandomState(None)

        out_list: List[pd.DataFrame] = []
        if not arr_train:
            return out_list

        targets_stat = []
        for df in arr_train:
            if target_col not in df.columns:
                targets_stat.append([0])
                continue

            col_data = df[target_col]
            stats = [
                col_data.mean(),
                col_data.std(),
                col_data.max(),
                col_data.min(),
                col_data.median(),
                col_data.sum(),
                col_data.kurtosis(),
                col_data.skew()
            ]
            targets_stat.append(stats)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(targets_stat)

        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(features_scaled)

        cluster_counts = pd.Series(clusters).value_counts()
        # print(f"聚类数量分布: {cluster_counts.to_dict()}")

        samples_count = cluster_counts.mean() # 聚类类别的样本数量均值作为每个类别保留的最大样本数
        # print(f"每个聚类将保留 {samples_count} 个样本")

        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
            if len(cluster_indices) > samples_count:
                selected_indices = np.random.choice(
                    cluster_indices,
                    size=round(samples_count),
                    replace=False
                )
            else:
                selected_indices = cluster_indices

            for idx in selected_indices:
                out_list.append(arr_train[idx])

        # print(f"处理前样本总数: {len(arr_train)}")
        # print(f"处理后样本总数: {len(out_list)}")

        return out_list


