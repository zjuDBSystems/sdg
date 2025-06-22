from enum import Enum
from dataclasses import dataclass
import json
import os
import shutil
from pprint import pprint
from uuid import uuid4

from .power_table_data.frequency import energy_ratio_all, spectral_entropy_all, snr_all
from .power_table_data.multi_scale import acf_multiscale_all, trend_strength_all, cross_scale_mi_all
from .power_table_data.non_stationary import adf_nonstationarity_all, rolling_var_cv_all, hurst_all
from .power_table_data.time import bucket_coverage_csv, holiday_balance_csv, season_span_csv, solar_angle_coverage_csv
from ..config import settings

from ..config import settings
from .image_code_data.data_size import calculate_score_from_csv
from .image_code_data.syntax import evaluate_js_folder
from .image_code_data.renderable import evaluate_renderability
from .image_code_data.ncc import evaluate_ncc
from .image_code_data.chart_type import evaluate_chart_type
from .image_code_data.option_diversity import evaluate_option_diversity
from .image_code_data.code_duplication import evaluate_code_duplicate
from .image_code_data.image_duplication import evaluate_image_duplicate
from .image_code_data.joint_duplicate import evaluate_joint_duplicate
from .image_code_data.config_complete import evaluate_completeness
from .image_code_data.missing_rate_detection import evaluate_miss

import pandas as pd

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

    def __init__(self, dirs: list[Datadir], meta_path: str,md_path:str):
        self.dirs = dirs
        self.relative_meta_path = meta_path
        self.meta_path = settings.LOCAL_META_STORAGE_PATH + '/' + meta_path
        screenshot_path = settings.LOCAL_STORAGE_PATH + '/screenshot'
        self.relative_md_path = md_path
        md_path = settings.LOCAL_META_STORAGE_PATH + '/' + md_path
        result_path = settings.LOCAL_STORAGE_PATH + '/result.csv'
        self.md_path = md_path
        self.type_percentage={}
        self.evaluation = {
            "screenshot_path": screenshot_path,
            "md_path": md_path,
            "result_path": result_path
        }

    def evaluate_table_quality(self):
        table_file_path = self.dirs[0].data_path  # 表格路径

        # 1、数据量的计算
        from .power_table_data.data_size import calculate_score_from_csv
        data_row_score, data_col_score = calculate_score_from_csv(table_file_path)

        # 2、主频得分的计算
        energy_ratio_ser = energy_ratio_all(table_file_path)
        spectral_entropy_ser = spectral_entropy_all(table_file_path)
        snr_ser = snr_all(table_file_path)

        energy_ratio_score = energy_ratio_ser.mean()
        spectral_entropy_score = spectral_entropy_ser.mean()
        snr_score = snr_ser.mean()

        # 3、非平稳得分的计算
        adf_pvalue_ser = adf_nonstationarity_all(table_file_path)
        rolling_var_cv_ser = rolling_var_cv_all(table_file_path)
        hurst_ser = hurst_all(table_file_path)

        adf_pvalue_score = adf_pvalue_ser.mean()
        rolling_var_cv_score = rolling_var_cv_ser.mean()
        hurst_exponent_score = hurst_ser.mean()

        # 时间特征得分计算
        temporal_bucket_coverage_score = bucket_coverage_csv(table_file_path)
        holiday_balance_score = holiday_balance_csv(table_file_path)
        season_span_score = season_span_csv(table_file_path)
        solar_angle_coverage_score = solar_angle_coverage_csv(table_file_path)

        # 多尺度得分计算
        acf_multiscale_ser = acf_multiscale_all(table_file_path)
        trend_strength_ser = trend_strength_all(table_file_path)
        cross_scale_mi_ser = cross_scale_mi_all(table_file_path)

        acf_multiscale_divergence_score = acf_multiscale_ser.mean()
        trend_strength_score = trend_strength_ser.mean()
        cross_scale_mi_score = cross_scale_mi_ser.mean()

        # 一级指标得分
        dominant_frequency_score = (energy_ratio_score + spectral_entropy_score + snr_score) / 3
        nonstationary_enhancement_score = (adf_pvalue_score + rolling_var_cv_score + hurst_exponent_score) / 3
        temporal_feature_enhancement_score = (temporal_bucket_coverage_score + holiday_balance_score + season_span_score + solar_angle_coverage_score) / 4
        multiscale_enhancement_score = (acf_multiscale_divergence_score + trend_strength_score + cross_scale_mi_score) / 3
        data_size_score = (data_row_score + data_col_score) / 2

        # 构建分数字典
        score_dict = {
            "一级指标": {
                "主频提取价值": round(dominant_frequency_score, 2),
                "非平稳增强价值": round(nonstationary_enhancement_score, 2),
                "时序特征增强价值": round(temporal_feature_enhancement_score, 2),
                "多尺度增强价值": round(multiscale_enhancement_score, 2),
                "数据量扩增价值": 100 - round(data_size_score, 2),
            },
            "二级指标": {
                "主频提取价值": {
                    "主频能量占比": round(energy_ratio_score, 2),
                    "谱熵": round(spectral_entropy_score, 2),
                    "信噪比": round(snr_score, 2),
                },
                "非平稳增强价值": {
                    "单位根检验": round(adf_pvalue_score, 2),
                    "滚动方差离散度": round(rolling_var_cv_score, 2),
                    "赫斯特指数": round(hurst_exponent_score, 2),
                },
                "时序特征增强必要性": {
                    "离散时间覆盖率": round(temporal_bucket_coverage_score, 2),
                    "工作日/节假日平衡": round(holiday_balance_score, 2),
                    "跨季节跨度": round(season_span_score, 2),
                    "太阳辐射角覆盖度": round(solar_angle_coverage_score, 2),
                },
                "多尺度增强价值": {
                    "多尺度自相关差异": round(acf_multiscale_divergence_score, 2),
                    "趋势强度": round(trend_strength_score, 2),
                    "跨尺度互信息": round(cross_scale_mi_score, 2),
                },
                "数据量扩增价值": {
                    "时间点得分": round(data_row_score, 2),
                    "特征得分": round(data_col_score, 2),
                }
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
        min_size = 550# 1k数据得0分
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
        chart_type_score,type_percentage = evaluate_chart_type(pair_file_path)
        chart_type_score = round(chart_type_score, 2)  # 保留两位小数
        self.type_percentage=type_percentage
        # 6、接着是配置项均衡性
        # print(2)
        option_diversity_score,file_to_cluster_center_distance = evaluate_option_diversity(code_file_path, pair_file_path)
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