'''Operators for syntax amend.
'''

from typing import override
import openai
import os
import pandas as pd
import json5
import json
from tqdm import tqdm
from llama_cpp import Llama
from ..config import settings

from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType

class SyntaxAmendOperatorGPT(Operator):
    def __init__(self, **kwargs):
        self.api_key = kwargs.get('api_key',"")
        self.model = kwargs.get('model', "gpt-4o")
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

            Field('api-key', Field.FieldType.STRING, 'OpenAI API key', ""),
            Field('model', Field.FieldType.STRING, 'OpenAI model name', "gpt-4o"),
            Field('score_file', Field.FieldType.STRING, 'Score result file path', "./detailed_scores.csv")
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
        
        # gpt-4o (github版)
        # client = openai.OpenAI(
        #     api_key = self.api_key,
        #     # base_url = "https://models.inference.ai.azure.com"
        #     base_url = settings.GPT_URL
        # )

        # 调用本地llm模型
        # llm = Llama(
        #     # model_path="./sdg/data_operator/model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        #     model_path="./sdg/data_operator/model/mistral-7b-instruct-v0.2.Q3_K_S.gguf",
        #     n_ctx=2048,
        #     n_threads=4,
        #     verbose=False,
        #     use_metal=True
        # )

        # 加载 0.4B 模型
        # model_path = "./sdg/data_operator/model/rwkv/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth"
        # # model_path ="./sdg/data_operator/model/rwkv/RWKV-4-Pile-3B-20221110-ctx4096.pth"
        # # tokenizer_path = "./sdg/data_operator/model/rwkv/20B_tokenizer.json"
        # tokenizer_path = "./sdg/data_operator/model/rwkv/rwkv_vocab_v20230424.json"
        # model = RWKV(model_path, strategy="cpu fp32")  
        # pipeline = PIPELINE(model, tokenizer_path)

        # 加载模型
        # MODEL_PATH = './sdg/data_operator/model/rwkv/RWKV-4-World-0.4B-v1-20230529-ctx4096.pth'
        # MODEL_TYPE = 'rwkv_vocab_v20230424'
        # model = RWKV(model=MODEL_PATH, strategy='cpu fp32')
        # pipeline = PIPELINE(model, MODEL_TYPE)

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

            # new_code_data = self.call_gpt4o(client, code_data)
            # new_code_data = self.fix_broken_syntax(code_data)
            # new_code_data = self.fix_by_llm(llm,code_data)
            # new_code_data = self.fix_by_rwkv(pipeline, code_data)
            new_code_data = self.fix_by_llm_chat(llm, code_data)

            if (new_code_data):
                print(f"成功修复{code_file_name}")
                with open(code_file_path, 'wb') as f:
                    f.write(new_code_data.encode('utf-8'))
            else:
                print(f"{code_file_name}未修复")
                
    
    @staticmethod
    def fix_by_rwkv(pipeline, bad_json):

        prompt = f"""### 指令:
        请修复以下无效的 JSON，并返回合法的 JSON，请只输出json代码！不需要描述与分析！

        {bad_json}

        ### 回答:"""
        response_text = pipeline.generate(prompt, token_count=500)
        # response_text = pipeline.run(prompt, max_tokens=300)
        print(f"得到响应{response_text}")
        start = response_text.find("{")
        end = response_text.rfind("}")
        return response_text[start:end+1]

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
    def fix_by_llm(llm, bad_json):

        prompt = f"""### 指令:
        请修复以下无效的 JSON，并返回合法的 JSON，请只输出json代码，不需要描述与分析：

        {bad_json}

        ### 回答:"""

        output = llm(prompt, max_tokens=300, temperature=0.2)
        print(f"获得修正结果{output}")
        response_text = output["choices"][0]["text"]
        start = response_text.find("{")
        end = response_text.rfind("}")
        return response_text[start:end+1]

    def call_gpt4o (self, client, code_data):

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                # {"role": "system", "content": "你是一个熟悉 ECharts 的前端开发专家"},
                {"role": "user", "content": "以下的echarts配置json代码无法编译成功，可能是语法存在错误，请对该json文件进行检测和修正。请只输出json代码，不需要描述与分析。"},
                {"role": "user", "content": code_data}
            ]
        )

        response_text = response.choices[0].message.content
        print("收到的结果为：" + response_text)
        start = response_text.find("{")
        end = response_text.rfind("}")
        return response_text[start:end+1]

        return json_text
    
    @staticmethod
    def fix_broken_syntax(json_str):
        try:
            # 尝试用 json5 解析（宽松模式）
            parsed_data = json5.loads(json_str)
            # 转回标准 JSON
            fixed_json = json.dumps(parsed_data, indent=2)
            
            return fixed_json
        except Exception as e:
            print(f"无法修复 JSON: {e}")
            return None

    @staticmethod
    def get_pending_files(csv_path, score_name, file_type):
        # 读取 CSV 文件（处理可能存在的空值）
        df = pd.read_csv(csv_path, na_values=['', ' ', 'NA'], dtype={score_name: float})

        # 筛选 syntax_score < 100 的行（自动排除 NaN 值）
        filtered_df = df[df[score_name] < 100]

        # 提取 code 字段并转换为列表
        code_list = filtered_df[file_type].dropna().tolist()  # 同时过滤 code 列可能的空值

        return code_list