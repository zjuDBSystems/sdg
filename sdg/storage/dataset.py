from enum import Enum
from dataclasses import dataclass
import os
import shutil
from uuid import uuid4

from ..config import settings

from ..config import settings
from .image_code_data.data_size import calculate_score_from_csv
from .image_code_data.syntax import evaluate_js_folder
from .image_code_data.renderable import evaluate_renderability
from .image_code_data.ssim import evaluate_ssim
from .image_code_data.chart_type import evaluate_chart_type
from .image_code_data.option_diversity import evaluate_option_diversity
from .image_code_data.code_duplication import evaluate_code_duplicate
from .image_code_data.image_duplication import evaluate_image_duplicate
from .image_code_data.joint_duplicate import evaluate_joint_duplicate
from .image_code_data.config_complete import evaluate_completeness

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
    ECHARTS = 'echarts'
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
        md_path = settings.LOCAL_META_STORAGE_PATH + '/' + md_path
        result_path = settings.LOCAL_STORAGE_PATH + '/result.csv'

        self.evaluation = {
            "screenshot_path": screenshot_path,
            "md_path": md_path,
            "result_path": result_path
        }


    def evaluate_image_code_quality(self):
        code_file_path = self.dirs[0].data_path  # 代码路径
        image_file_path = self.dirs[1].data_path  # 图像路径
        pair_file_path = self.meta_path  # 图像代码配对文件路径
        screenshot_path = self.evaluation["screenshot_path"]
        md_path = self.evaluation["md_path"]
        result_path = self.evaluation["result_path"]
        print(screenshot_path)
        print(md_path)
        print(result_path)
        # 最终的结果文档创造d
        collector = ScoreCollector(pair_file_path)

        # 首先是数据量的计算
        min_size = 1000  # 1k数据得0分
        max_size = 10 ** 5  # 100K数据得100分
        data_size_score = calculate_score_from_csv(pair_file_path, min_size, max_size)
        print("dataset_score:",data_size_score)
        # 接下来是代码质量的计算
        # 首先是语法检测
        syntax_score, syntax_score_details = evaluate_js_folder(code_file_path)
        collector.add_scores("syntax_score", syntax_score_details)
        print("syntax_score:",syntax_score)
        # print(0)
        # 接着是可渲染性检测
        print('start renderable test')
        renderable_score, renderable_score_details = evaluate_renderability(code_file_path,
                                                                            screenshot_path)  # 文件内部需要提供一个浏览器的所在地址
        collector.add_scores("renderable_score", renderable_score_details)
        print("renderable_score:",renderable_score)
        print(renderable_score)
        # 之后是配置项完整性检测
        configuration_complete_score, configuration_complete_score_details = evaluate_completeness(md_path,
                                                                                                   pair_file_path,
                                                                                                   code_file_path)
        collector.add_scores("configuration_complete_score", configuration_complete_score_details)
        print("configuration_complete_score:",configuration_complete_score)
        # print("配置项完整")
        # print(configuration_complete_score)
        # 接下来是图像代码对齐
        # 首先是原图像与截图的SSIM
        ssim_score, ssim_score_details = evaluate_ssim(pair_file_path, image_file_path, screenshot_path)#不能有中文路径

        collector.add_scores("ssim_score", ssim_score_details, key_type="image")
        print("ssim_score:",ssim_score)


        # print(ssim_score)



        # 接下来是数据集多样性
        # 首先是图表类型均衡性
        # print(1)
        chart_type_score = evaluate_chart_type(pair_file_path)
        print("chart_type_score:",chart_type_score)
        # 接着是配置项均衡性
        # print(2)
        option_diversity_score = evaluate_option_diversity(code_file_path, pair_file_path)
        print("option_diversity_score:",option_diversity_score)
        # print(option_diversity_score)
        # 接下来是重复检测
        # 首先是代码重复检测
        # print(3)
        code_duplicate_score, duplicate_code_files = evaluate_code_duplicate(code_file_path)
        print("code_duplicate_score:",code_duplicate_score)
        # print(code_duplicate_score)
        # print(duplicate_code_files)
        # 接下来是图像重复检测
        # print(4)
        image_duplicate_score, duplicate_image_files = evaluate_image_duplicate(image_file_path)
        print("image_duplicate_score:",image_duplicate_score)
        # print(image_duplicate_score)
        # print(duplicate_image_files)
        # 之后是联合重复检测
        # print(5)
        joint_duplicate_score, duplicate_joint_files = evaluate_joint_duplicate(duplicate_code_files,
                                                                                duplicate_image_files, pair_file_path)
        print("joint_duplicate_score:",joint_duplicate_score)
        # print(joint_duplicate_score)
        # print(duplicate_joint_files)
        # print(6)
        collector.add_exclusion_list('code_duplicate_score', duplicate_code_files, key_type='code')
        collector.add_exclusion_list('image_duplicate_score', duplicate_image_files, key_type='image')
        collector.add_exclusion_list('joint_duplicate_score', duplicate_joint_files, key_type='code')
        # print(7)

        # 一级指标得分

        code_quality_score = (syntax_score + renderable_score + configuration_complete_score) / 3
        image_code_alignment_score = ssim_score
        dataset_diversity_score = (chart_type_score + option_diversity_score) / 2
        data_repeatability_score = (code_duplicate_score + image_duplicate_score + joint_duplicate_score) / 3

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
                    "可渲染性检测": renderable_score,
                    # "配置项完整检测": configuration_complete_score
                },
                "图像代码对齐": {
                    "图像与渲染截图的SSIM": ssim_score

                },
                "数据集多样性": {
                    "图表类型均衡性": chart_type_score,
                    "配置项多样性": option_diversity_score
                },
                "数据重复性": {
                    "代码重复": code_duplicate_score,
                    "图像重复": image_duplicate_score,
                    "联合重复": joint_duplicate_score
                },
                "数据量": {
                    "数据量": data_size_score
                }
            }
        }

        # 结果文档
        final_df = collector.generate_report(result_path)#需要填写结果文档所处位置
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
    dataset: Dataset = Dataset(dirs, meta_path)
    shutil.copy(src.meta_path, dataset.meta_path)
    return dataset