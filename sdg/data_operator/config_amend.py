'''Operators for syntax amend.
'''

from typing import override, Dict
import os
import pandas as pd
import json

from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType

class ConfigAmendOperator(Operator):
    def __init__(self, **kwargs):
        self.score_file = kwargs.get('score_file', "./detailed_scores.csv")

    @classmethod
    @override
    def accept(cls, data_type, task_type) -> bool:
        if data_type == DataType.CODE and task_type == TaskType.AUGMENTATION:
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
            name='ConfigAmendOperator',
            description='Config amend.'
        )
    
    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "ConfigAmendOperator"
        # records count
        poc_files = self.get_pending_files(self.score_file, 'configuration_complete_score', 'code')
        cost["ri"] = len(poc_files)
        # time of one record
        cost["ti"] = 0.0031
        # cpi time of one record
        cost["ci"] = 0.0019
        # operator type
        cost["type"] = "CPU"
        return cost

    @override
    def execute(self, dataset):
        

        # files
        code_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.CODE][0]
        # code_files = ['sqaure_pie_chart_2.json','sqaure_pie_chart_3.json']
        poc_files = self.get_pending_files(self.score_file, 'configuration_complete_score', 'code')
        print(f'修复的记录数为{len(poc_files)}')

        for index, (code_file_name,chart_type) in enumerate(poc_files):
            
            if pd.isna(code_file_name):
                continue

            code_file_path = os.path.join(code_dir.data_path,code_file_name)

            with open(code_file_path, 'rb') as f:
                code_data = f.read().decode('utf-8')

            new_code_data = self.fix_config(code_data, chart_type)


            with open(code_file_path, 'wb') as f:
                f.write(new_code_data.encode('utf-8'))
            

    # 更新 series 配置函数
    @staticmethod
    def update_series_config(series_item, type):
        if (type == "half_doughnut"):
            series_item["type"] = "pie"
            series_item["startAngle"] = 180
            series_item["endAngle"] = 360
        elif (type == "pie_chart"):
            series_item["type"] = "pie"
        elif (type == "square_pie"):
            series_item["type"] = "pie"
            series_item["startAngle"] = 90
            series_item["endAngle"] = 360
        elif (type == "tangential_polar_bar"):
            series_item["type"] = "bar"
            series_item["coordinateSystem"] = 'polar'
        elif (type == "tangential_polar_bar-radiusaxis"):
            series_item["type"] = "category"

    def fix_config(self, coda_data, chart_type):

        # print(f"开始修复的图表类型为{chart_type}")

        # 转为dict
        data_dict = json.loads(coda_data)

        half_doughnut = "half_doughnut".strip().lower()
        pie_chart = "pie_chart".strip().lower()
        square_pie = "square_pie".strip().lower()
        Tangential_Polar_Bar = "Tangential_Polar_Bar".strip().lower()

        chart_type = chart_type.strip().lower()
        # 处理series
        if chart_type in (half_doughnut,pie_chart,square_pie,Tangential_Polar_Bar):
            # 如果 series 不存在，则添加一个默认的 dict
            if "series" not in data_dict or data_dict["series"] is None:
                data_dict["series"] = [{}]  # 默认加一个空 dict 再修改
                self.update_series_config(data_dict["series"][0], chart_type)
            # list
            elif isinstance(data_dict["series"], list):
                if len(data_dict["series"]) == 0:
                    data_dict["series"].append({})
                if isinstance(data_dict["series"][0], dict):
                    self.update_series_config(data_dict["series"][0], chart_type)
                else:
                    print("series[0] 不是 dict，跳过处理")
            # dict
            elif isinstance(data_dict["series"], dict):
                self.update_series_config(data_dict["series"], chart_type)
            else:
                print("series 格式不支持：既不是列表也不是字典")
        # 处理Tangential_Polar_Bar的radiusAxis
        elif chart_type in (Tangential_Polar_Bar):
            # 如果 radiusAxis 不存在，则添加一个默认的 dict
            if "radiusAxis" not in data_dict or data_dict["radiusAxis"] is None:
                data_dict["radiusAxis"] = {}  # 默认加一个空 dict 再修改
                self.update_series_config(data_dict["radiusAxis"], chart_type+"-radiusaxis")
            # dict
            elif isinstance(data_dict["radiusAxis"], dict):
                self.update_series_config(data_dict["radiusAxis"], chart_type+"-radiusaxis")
            else:
                print("radiusAxis 格式不支持")

        # 返回json字符串
        return json.dumps(data_dict, ensure_ascii=False, indent=2)


    @staticmethod
    def get_pending_files(csv_path, score_name, file_type):
        # 读取 CSV 文件（处理可能存在的空值）
        df = pd.read_csv(csv_path, na_values=['', ' ', 'NA'], dtype={score_name: float})

        # 筛选 syntax_score < 100 的行（自动排除 NaN 值）
        filtered_df = df[(df[score_name] < 90) & (df['type'].notna()) & (df['code'].notna())]

        # 提取 code 和 type 字段并转换为列表
        code_list = filtered_df[file_type]
        type_list = filtered_df['type']

        # 组合为 (code, type) 元组列表
        result_list = list(zip(code_list, type_list))

        return result_list