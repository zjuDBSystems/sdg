'''Operators for syntax amend.
'''

from typing import override
import os
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama
from transformers import AutoTokenizer
from ..config import settings

from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType

class SyntaxAmendOperator(Operator):
    def __init__(self, **kwargs):

        self.score_file = kwargs.get('score_file', "./scores.csv")

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

            Field('score_file', Field.FieldType.STRING, 'Score result file path', "./scores.csv")
        ]
    

    @classmethod
    @override
    def get_meta(cls) -> Meta:
        return Meta(
            name='SyntaxAmendOperator',
            description='Synmax amend.'
        )
    
    @override
    def execute(self, dataset):

        # 加载qwen模型
        llm = Llama(
            model_path="./sdg/data_operator/model/qwen1_5-0_5b-chat-q4_k_m.gguf",
            n_ctx=2048,  # 上下文窗口大小，适合小模型
            n_threads=4  # 设置为你的 CPU 核心数
        )

        # files
        code_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.CODE][0]
        # code_files = ['half_doughnut_chart_1.json','square_pie_chart_1.json']
        code_files = self.get_pending_files(self.score_file, 'syntax_score', 'code')

        for index, code_file_name in enumerate(tqdm(code_files, desc="修复进度")):
            
            if pd.isna(code_file_name):
                continue

            code_file_path = os.path.join(code_dir.data_path,code_file_name)

            with open(code_file_path, 'rb') as f:
                code_data = f.read().decode('utf-8')

            new_code_data = self.fix_by_llm_chat(llm, code_data)

            if (new_code_data):
                print(f"成功修复{code_file_name}")
                with open(code_file_path, 'wb') as f:
                    f.write(new_code_data.encode('utf-8'))
            else:
                print(f"{code_file_name}未修复")
                

    @staticmethod
    def fix_by_llm_chat(llm, bad_json):

        messages = [
            {"role": "system", "content": "你是一个echarts专家，简洁、有逻辑地回答问题。"},
            {"role": "user", "content": f"请修复以下无效的echarts配置的JSON，并返回合法的echarts配置JSON，请只输出json代码，不需要描述与分析：{bad_json}"}
        ]

        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")
        # tokens = tokenizåer(str(bad_json), add_special_tokens=False)
        # input_tokens_len = len(tokens['input_ids'])
        # print(f"Token 数量：{input_tokens_len}")

        response = llm.create_chat_completion(messages=messages, max_tokens=512)
        output = response['choices'][0]['message']['content']
        # print("Completion tokens:", response['usage']['completion_tokens'])
        # print(f"修复前后tokens差值{response['usage']['completion_tokens']-input_tokens_len}")
        # print(f"获得修正结果{output}")
        start = output.find("{")
        end = output.rfind("}")
        return output[start:end+1]

    @staticmethod
    def get_pending_files(csv_path, score_name, file_type):
        # 读取 CSV 文件（处理可能存在的空值）
        df = pd.read_csv(csv_path, na_values=['', ' ', 'NA'], dtype={score_name: float})

        # 筛选 syntax_score < 100 的行（自动排除 NaN 值）
        filtered_df = df[df[score_name] < 100]

        # 提取 code 字段并转换为列表
        code_list = filtered_df[file_type].dropna().tolist()  # 同时过滤 code 列可能的空值

        return code_list