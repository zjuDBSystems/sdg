'''Operators for syntax amend.
'''

from typing import override
import openai
import os
import pandas as pd
from ..config import settings

from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType

class ConfigAmendOperator(Operator):
    def __init__(self, **kwargs):
        self.api_key = kwargs.get('api_key',"")
        self.model = kwargs.get('model', "gpt-4o")

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

            Field('api-key', Field.FieldType.STRING, 'OpenAI API key', ""),
            Field('model', Field.FieldType.STRING, 'OpenAI model name', "gpt-4o")
        ]
    

    @classmethod
    @override
    def get_meta(cls) -> Meta:
        return Meta(
            name='ConfigAmendOperator',
            description='Config amend.'
        )
    
    @override
    def execute(self, dataset):
        
        # gpt-4o (github版)
        client = openai.OpenAI(
            api_key = self.api_key,
            # base_url = "https://models.inference.ai.azure.com"
            base_url = settings.GPT_URL
        )

        # files
        code_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.CODE][0]
        # code_files = ['sqaure_pie_chart_2.json','sqaure_pie_chart_3.json']
        poc_files = self.get_pending_files('./scores.csv', 'configuration_complete_score', 'code')

        key_config_path = "./metadata/key_configurations.md"
        with open(key_config_path, 'r', encoding='utf-8') as file:
            key_config = file.read()
        

        for index, (code_file_name,chart_type) in enumerate(poc_files):
            
            if pd.isna(code_file_name):
                continue

            code_file_path = os.path.join(code_dir.data_path,code_file_name)

            with open(code_file_path, 'rb') as f:
                code_data = f.read().decode('utf-8')

            new_code_data = self.call_gpt4o(client, code_data, chart_type, key_config)

            with open(code_file_path, 'wb') as f:
                f.write(new_code_data.encode('utf-8'))
            

    def call_gpt4o (self, client, code_data, chart_type, config):

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                # {"role": "system", "content": "你是一个熟悉 ECharts 的前端开发专家"},
                {"role": "user", "content": f"以下是echarts配置json代码，以及echarts不同类型代码的关键配置文档。echarts的json代码的配置项可能存在一定程度的缺失，以及具体取值与相应配置项不匹配的情况，需要对该echarts配置进行修改。该json配置的echarts类型为{chart_type},在关键配置文档中找到相应类型，将不符合该文档中相应配置取值的配置，改为要求的取值。请只输出json代码，不需要描述与分析。"},
                {"role": "user", "content": code_data},
                {"role": "user", "content": config}

            ]
        )

        response_text = response.choices[0].message.content
        print("收到的结果为：" + response_text)
        start = response_text.find("{")
        end = response_text.rfind("}")
        json_text = response_text[start:end+1]

        return json_text

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