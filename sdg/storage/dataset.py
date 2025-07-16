import os
import pickle as pkl
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from uuid import uuid4
from typing import List
import pandas as pd

from .image_code_data.chart_type import evaluate_chart_type
from .image_code_data.code_duplication import evaluate_code_duplicate
from .image_code_data.config_complete import evaluate_completeness
from .image_code_data.data_size import calculate_score_from_csv
from .image_code_data.image_duplication import evaluate_image_duplicate
from .image_code_data.missing_rate_detection import evaluate_miss
from .image_code_data.ncc import evaluate_ncc
from .image_code_data.option_diversity import evaluate_option_diversity
from .image_code_data.syntax import evaluate_js_folder
from .power_table_data.data_context_quality import score_calculate_domain_diversity, score_calculate_domain_completeness
from .power_table_data.data_inner_quality import score_missing_rate, score_label_consistency
from .power_table_data.data_redundancy import score_feature_independence
from .power_table_data.data_representation_quality import score_stationarity_all, score_feature_readiness
from .power_table_data.data_size import score_time_granularity, score_seasonality_strength, score_trend_strength, \
    score_primary_freq_strength, score_dataset_balance
from ..config import settings


class DataType(Enum):
    """Available data types."""
    PYTHON = 'python'
    IMAGE = 'image'
    AUDIO = 'audio'
    TEXT = 'text'
    VIDEO = 'video'
    TABLE = 'table'
    GRAPH = 'graph'
    # ECHARTS = 'echarts'
    CODE = 'code'


class ScoreCollector:
    def __init__(self, pair_file):
        self.df = pd.read_csv(pair_file)
        self.scores = {
            # 'renderable': {},
            # 'syntax': {},
            # 'completeness': {},
            # # 添加其他指标...
        }
        self.missing_data = {}

    def add_missing_data(self, miss_dict):
        """
        将缺失的数据类型添加到 missing_data 字典
        :param miss_dict: 缺失数据字典，包含行索引和缺失类型（'code'、'image'等）
        """
        self.missing_data = miss_dict

    def add_exclusion_list(self, metric_name: str, exclusion_list: list, key_type: str = 'code'):
        """
        添加排除列表评分规则（出现在列表中的得0分，其他得100分）

        参数:
            metric_name: 指标名称 (例如 'valid_images')
            exclusion_list: 需要排除的文件名列表
            key_type: 键类型 ('code'=代码文件 / 'image'=图像文件)
        """
        if key_type not in ['code', 'image']:
            raise ValueError("key_type 必须是 'code' 或 'image'")

        # 确定映射列
        map_column = 'code' if key_type == 'code' else 'image'

        # 生成评分字典（默认100分）
        scores_dict = {item: 100 for item in self.df[map_column].unique()}

        # 处理排除项
        for item in exclusion_list:
            if item in scores_dict:
                scores_dict[item] = 0
            else:
                print(f"警告：排除项 {item} 不存在于 {map_column} 列")

        # 添加到评分体系
        self.add_scores(metric_name, scores_dict, key_type)

    def add_scores(self, metric_name: str, scores_dict: dict, key_type: str = 'code'):
        """
        添加评分结果
        :param metric_name: 指标名称 (e.g. 'syntax')
        :param scores_dict: 评分字典，键为 code_file 或 image_file
        :param key_type: 键类型，'code' 或 'image'
        """
        if key_type not in ['code', 'image']:
            raise ValueError("key_type 必须是 'code' 或 'image'")
        self.scores[metric_name] = (scores_dict, key_type)

    def generate_report(self, output_file):
        """生成最终报告CSV"""
        # 遍历所有评分指标
        for metric, (scores_dict, key_type) in self.scores.items():
            # 根据键类型选择映射列
            map_column = 'code' if key_type == 'code' else 'image'
            self.df[metric] = self.df[map_column].map(scores_dict)
        self.df['missing_data'] = self.df.index.map(lambda idx: ','.join(self.missing_data.get(idx, ['perfect'])))
        # 保存并返回结果
        self.df.to_csv(output_file, index=False)
        return self.df


@dataclass
class DataEvaluation():
    # evaluation dimension key
    key: str
    # evaluation value
    value: str
    # evaluation result path
    detail_path: str
    # sub dimension evaluations
    children: list['DataEvaluation']


class Datadir:
    """A datadir is a collection of data files with the same type.

    Attributes:
        data_path (str): The path to the data dir.
        data_type (DataType): The type of the data files.
    """

    def __init__(self, data_path: str, data_type: DataType):
        self.relative_data_path = data_path
        self.data_path = settings.LOCAL_STORAGE_PATH + '/' + data_path
        self.data_type = data_type


class Dataset:
    """A dataset is a collection of datadirs.

    Note: The file format for the metadata is CSV, the header include data type(exists multi data-type if dataset 
    is multimodal data) and tags(split by |), join data_path in datadir and file name in metadata could touch the
    file.

    Examples:
    1. normal dataset
    code,tags
    1.py,tag1
    2.py,tag2|tag3
    2. multimodal dataset
    code,image,tags
    1.py,1.jpg,tag1
    1.py,2.jpg,tag1
    2.py,3.jpg,tag2
    
    
    Attributes:
        dirs (list[Datadir]): A list of datadirs.
        meta_path (str): The path to the metadata file, which connect mutlimodal data.
        evaluation (list[DataEvaluation]): A list of data evaluation result.
    """

    def __init__(self, dirs: list[Datadir], meta_path: str, md_path: str):
        self.dirs = dirs
        self.relative_meta_path = meta_path
        self.meta_path = settings.LOCAL_META_STORAGE_PATH + '/' + meta_path
        screenshot_path = settings.LOCAL_STORAGE_PATH + '/screenshot'
        self.relative_md_path = md_path
        md_path = settings.LOCAL_META_STORAGE_PATH + '/' + md_path
        result_path = settings.LOCAL_STORAGE_PATH + '/result.csv'
        self.md_path = md_path
        self.type_percentage = {}
        self.evaluation = {
            "screenshot_path": screenshot_path,
            "md_path": md_path,
            "result_path": result_path
        }

    def evaluate_table_quality(self, path="shanxi_day_train_total_96_96.pkl"):
        table_file_path = os.path.join(self.dirs[0].data_path, path)
        arr_evaluation = pkl.load(open(table_file_path, "rb"))

        def restore_datetime_index_to_column(
                df_list: List[pd.DataFrame],
                col_name: str = "datetime") -> List[pd.DataFrame]:
            restored_list = []
            for df in df_list:
                df_copy = df.copy()
                if col_name not in df_copy.columns:
                    df_copy = df_copy.rename_axis(col_name).reset_index()
                restored_list.append(df_copy)
            return restored_list       

        arr_evaluation = restore_datetime_index_to_column(arr_evaluation) 

        target_col = "延安发电1号机组"

        ep_col = [
            '延安发电1号机组', '延安发电2号机组', '延热发电1号机组', '延热发电2号机组', '宝二发电1号机组',
            '宝二发电2号机组',
            '宝二发电3号机组', '宝二发电4号机组', '宝鸡发电5号机组', '宝鸡发电6号机组', '宝热发电1号机组',
            '宝热发电2号机组',
            '杨凌热电1号机组', '杨凌热电2号机组', '彬长发电1号机组', '彬长发电2号机组', '灞桥发电1号机组',
            '灞桥发电2号机组',
            '富平发电1号机组', '富平发电2号机组', '韩二发电1号机组', '韩二发电2号机组', '韩二发电3号机组',
            '韩二发电4号机组',
            '蒲城发电3号机组', '蒲城发电4号机组', '蒲二发电5号机组', '蒲二发电6号机组', '秦岭发电7号机组',
            '秦岭发电8号机组',
            '渭河热电1号机组', '渭河热电2号机组', '渭南发电1号机组', '渭南发电2号机组', '西热发电1号机组', '西热发电2号机组'
        ]

        vote_cols = ['延安发电1号机组', '延安发电2号机组', '延热发电1号机组', '延热发电2号机组']

        EXPECTED_TIME_FEATURES = [
            'month', 'day', 'weekday', 'hour', 'minute', 'holiday', 'elev', 'az'
        ]

        api_key = "sk-2694f692c8a74876a7a8856fdaf7ed7e"

        time_granularity_score = score_time_granularity(arr_evaluation)
        seasonality_strength_score = score_seasonality_strength(arr_evaluation, method='classical')
        trend_strength_score = score_trend_strength(arr_evaluation, method='classical')
        primary_freq_strength_score = score_primary_freq_strength(arr_evaluation)
        dataset_balance_score = score_dataset_balance(arr_evaluation, value_cols=[target_col])

        missing_rate_score = score_missing_rate(arr_evaluation)
        label_consistency_score = score_label_consistency(arr_evaluation, label_col=target_col, vote_cols=vote_cols)

        stationarity_score = score_stationarity_all(arr_evaluation)
        temporal_features_ratio_score = score_feature_readiness(arr_evaluation, target_col=target_col,
                                                                expected_features=EXPECTED_TIME_FEATURES)

        calculate_domain_diversity_score = score_calculate_domain_diversity(arr_evaluation, api_key)
        calculate_domain_completeness_score = score_calculate_domain_completeness(arr_evaluation, api_key)

        all_cols = arr_evaluation[0].columns.tolist()
        cols_to_keep = [
            col for col in all_cols
            if not any(re.search(pattern, col, re.IGNORECASE) for pattern in ep_col)
        ]
        feature_independence_score = score_feature_independence(
            arr_evaluation, target_col=target_col, remain_cols=cols_to_keep)

        # 构建分数字典
        score_dict = {
            "一级指标": {
                "数据量": (time_granularity_score + seasonality_strength_score +
                           trend_strength_score + primary_freq_strength_score +
                           dataset_balance_score) / 5,
                "数据内在质量": (missing_rate_score + label_consistency_score) / 2,
                "数据表示质量": (stationarity_score + temporal_features_ratio_score) / 2,
                "数据上下文质量": (calculate_domain_diversity_score +
                                   calculate_domain_completeness_score) / 2,
                "数据冗余": (dataset_balance_score + feature_independence_score) / 2,
            },
            "二级指标": {
                "数据量": {
                    "时间粒度覆盖率": time_granularity_score,
                    "季节性强度": seasonality_strength_score,
                    "趋势强度": trend_strength_score,
                    "主频强度": primary_freq_strength_score,
                    "样本均衡性": dataset_balance_score,
                },
                "数据内在质量": {
                    "数据完整性": missing_rate_score,
                    "标签一致性": label_consistency_score,
                },
                "数据表示质量": {
                    "时序平稳性": stationarity_score,
                    "时间特征完备度": temporal_features_ratio_score,
                },
                "数据上下文质量": {
                    "领域知识多样性": calculate_domain_diversity_score,
                    "领域知识完整性": calculate_domain_completeness_score,
                },
                "数据冗余": {
                    "样本均衡性": dataset_balance_score,
                    "特征独立性": feature_independence_score,
                },
            }
        }

        pprint(score_dict, width=80, compact=True)

        return score_dict

    def evaluate_image_code_quality(self):
        code_file_path = self.dirs[0].data_path  # 代码路径
        image_file_path = self.dirs[1].data_path  # 图像路径
        pair_file_path = self.meta_path  # 图像代码配对文件路径
        screenshot_path = self.evaluation["screenshot_path"]
        md_path = self.evaluation["md_path"]
        result_path = self.evaluation["result_path"]
        # print(screenshot_path)
        # print(md_path)
        # print(result_path)
        # 最终的结果文档创造d
        collector = ScoreCollector(pair_file_path)

        # 1、首先是数据量的计算
        min_size = 550  # 1k数据得0分
        max_size = 800  # 100K数据得100分
        data_size_score = calculate_score_from_csv(pair_file_path, min_size, max_size)
        data_size_score = round(data_size_score, 2)  # 保留两位小数

        # 2、接下来是代码质量的计算
        # 首先是语法检测
        syntax_score, syntax_score_details = evaluate_js_folder(code_file_path)
        syntax_score = round(syntax_score, 2)  # 保留两位小数
        collector.add_scores("syntax_score", syntax_score_details)
        # 接着是可渲染性检测

        # 3、之后是配置项完整性检测
        configuration_complete_score, configuration_complete_score_details = evaluate_completeness(md_path,
                                                                                                   pair_file_path,
                                                                                                   code_file_path)
        configuration_complete_score = round(configuration_complete_score, 2)  # 保留两位小数
        collector.add_scores("configuration_complete_score", configuration_complete_score_details)
        # 接下来是图像代码对齐
        # 4、首先是原图像与截图的匹配度（NCC归一化互相关指数）
        ncc_score, ncc_score_details = evaluate_ncc(pair_file_path, image_file_path, screenshot_path, code_file_path)
        ncc_score = round(ncc_score, 2)  # 保留两位小数
        collector.add_scores("ncc_score", ncc_score_details, key_type="image")
        # 9、缺失率
        miss_score, missing_dict = evaluate_miss(pair_file_path)
        miss_score = round(float(miss_score), 2)  # 保留两位小数
        collector.add_missing_data(missing_dict)

        # 接下来是数据集多样性
        # 5、首先是图表类型均衡性
        # print(1)
        chart_type_score, type_percentage = evaluate_chart_type(pair_file_path)
        chart_type_score = round(chart_type_score, 2)  # 保留两位小数
        self.type_percentage = type_percentage
        # 6、接着是配置项均衡性
        # print(2)
        option_diversity_score, file_to_cluster_center_distance = evaluate_option_diversity(code_file_path,
                                                                                            pair_file_path)
        option_diversity_score = round(float(option_diversity_score), 2)  # 保留两位小数
        collector.add_scores("distance", file_to_cluster_center_distance, key_type="code")

        # 接下来是重复检测
        # 7、首先是代码重复检测
        # print(3)
        code_duplicate_score, duplicate_code_files = evaluate_code_duplicate(code_file_path)
        code_duplicate_score = round(code_duplicate_score, 2)  # 保留两位小数
        # 8、接下来是图像重复检测
        # print(4)
        image_duplicate_score, duplicate_image_files = evaluate_image_duplicate(image_file_path)
        image_duplicate_score = round(image_duplicate_score, 2)  # 保留两位小数

        collector.add_exclusion_list('code_duplicate_score', duplicate_code_files, key_type='code')
        collector.add_exclusion_list('image_duplicate_score', duplicate_image_files, key_type='image')

        # collector.add_exclusion_list('joint_duplicate_score', duplicate_joint_files, key_type='code')
        # print(7)

        # 一级指标得分
        # 一级指标得分
        code_quality_score = round((syntax_score + configuration_complete_score) / 2, 2)
        image_code_alignment_score = round((ncc_score + miss_score) / 2, 2)
        dataset_diversity_score = round((chart_type_score + option_diversity_score) / 2, 2)
        data_repeatability_score = round((code_duplicate_score + image_duplicate_score) / 2, 2)

        # 构建分数字典
        score_dict = {
            "一级指标": {
                "代码质量": code_quality_score,
                "图像代码对齐": image_code_alignment_score,
                "数据集多样性": dataset_diversity_score,
                "数据重复性": data_repeatability_score,
                "数据量": data_size_score
            },
            "二级指标": {
                "代码质量": {
                    "语法检测": syntax_score,
                    "配置项完整检测": configuration_complete_score
                },
                "图像代码对齐": {
                    "图像与渲染截图的匹配度": ncc_score,
                    "缺失率得分": miss_score
                },
                "数据集多样性": {
                    "图表类型均衡性": chart_type_score,
                    "配置项多样性": option_diversity_score
                },
                "数据重复性": {
                    "代码重复": code_duplicate_score,
                    "图像重复": image_duplicate_score
                },
                "数据量": {
                    "数据量": data_size_score
                }
            }
        }

        # 结果文档
        final_df = collector.generate_report("./detailed_scores.csv")
        # print(final_df)
        return score_dict


def copy_dataset(src: Dataset):
    dirs: list[Datadir] = []
    for dir in src.dirs:
        relative_data_path = str(uuid4())
        data_type = dir.data_type
        target_dir = Datadir(relative_data_path, data_type)
        # os.mkdir(target_dir.data_path)
        shutil.copytree(dir.data_path, target_dir.data_path)
        dirs.append(target_dir)
    meta_path = str(uuid4()) + '.metadata'
    dataset: Dataset = Dataset(dirs, meta_path, src.relative_md_path)
    shutil.copy(src.meta_path, dataset.meta_path)
    return dataset
