""" Operator for echarts converting
"""

from typing import override, Dict
import openai
from PIL import Image
import io
import os
import base64
import tiktoken
import pandas as pd
from tqdm import tqdm

from ..config import settings
from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType

class ImgToEchartsOperator(Operator):
    """ImgToEchartsOperator is an operator that converts an image dataset to an ECharts code dataset."""

    def __init__(self, **kwargs):
        self.api_key = kwargs.get('api_key',"")
        self.model = kwargs.get('model', "gpt-4o")

        self.token_count = 0  # 用于存储每次调用的token统计信息
        self.token_encoder = tiktoken.encoding_for_model("gpt-4")  # 使用tiktoken进行token计数

    @classmethod
    @override
    def accept(cls, data_type, task_type) -> bool:
        if data_type == DataType.IMAGE and task_type == TaskType.AUGMENTATION:
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
            name='ImgToEchartsOperator',
            description='Converts an image dataset to an ECharts code dataset.'
        )
    
    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "ImgToEchartsOperator"
        # records count
        cost["ri"] = self.get_record_count(dataset.meta_path)
        # time of one record
        cost["ti"] = 7.61
        # cpi time of one record
        input_token = 165
        cost["intoken"] = input_token
        output_token = 340
        cost["outtoken"] = output_token
        cost["ci"] = round( (input_token+output_token*4)*0.000018125 , 4)
        # operator type
        cost["type"] = "LLM"
        return cost

    def count_tokens(self, text):
        """使用tiktoken计算文本的token数量"""
        return len(self.token_encoder.encode(text))

    @override
    def execute(self, dataset):
        
        print(f"需要处理的数据量为{self.get_record_count(dataset.meta_path)}")

        # gpt-4o (github版)
        client = openai.OpenAI(
            api_key = self.api_key,
            # base_url = "https://models.inference.ai.azure.com"
            base_url = settings.GPT_URL
        )

        # files
        df = pd.read_csv(dataset.meta_path)
        img_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.IMAGE][0]
        code_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.CODE][0]
        img_files = df[DataType.IMAGE.value].tolist()
        code_files = df[DataType.CODE.value].tolist()

        # process
        for index, img_file_name in enumerate(tqdm(img_files, desc="修复进度")):
            # print("img_file : " + img_file_name)
            # For images without corresponding echarts code files
            if pd.isna(img_file_name):
                continue
            code_file_name = os.path.splitext(img_file_name)[0]+'.json'
            # print("code_file : " + code_file_name)
            if not self.check_file_existence(code_file_name, code_files):
                # Generate echarts code files
                img_file_path = os.path.join(img_dir.data_path, img_file_name)
                with open(img_file_path,'rb+') as image_f :
                    image=image_f.read()
                answer_content = self.call_gpt4o(client, image)
                # try:
                #     answer_content = self.call_gpt4o(client, image, 90)
                # except Exception as e:
                #     print(f"调用超时")
                #     # 异常时至少保留参数变异结果
                #     continue
                if not answer_content :
                    print('调用api获取数据失败')
                else :
                    code_file_path = os.path.join(code_dir.data_path, code_file_name)
                    
                    with open(code_file_path, 'wb') as code_f:
                        code_f.write(answer_content.encode('utf-8'))
                
                # modify csv file
                df.at[index, DataType.CODE.value] = code_file_name

        # save the modified csv
        df.to_csv(dataset.meta_path, index=False)
        print(f"处理的总token数{self.token_count}")

    # 减小图片的分辨率
    @staticmethod
    def compress_image_to_low_res(img_data: bytes) -> bytes:
        """将图片压缩至低分辨率（短边≤512px），返回压缩后的 bytes"""
        img = Image.open(io.BytesIO(img_data))
        
        # 计算缩放比例（保持宽高比）
        max_size = 512
        width, height = img.size
        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 保存为字节流（PNG格式）
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="PNG")
        return output_buffer.getvalue()
    
    def call_gpt4o (self, client, img_data):

        text_prompt = "Compose the ECharts code to achieve the same design and content as this chart screenshot.\n\n## Cautions\n- Write the json with ECharts directly. No html in the code.\n- Make sure the rendered chart looks exactly the same as the given image.\n- DO NOT miss the legend if it is in the image.\n- Just output the json code without description and analysis.\n\nLet's begin!\n"

        compressed_img_data = self.compress_image_to_low_res(img_data)

        # text_tokens = self.count_tokens(text_prompt)
        # image_tokens = 85 # 压缩后的图片为低分辨率的图片，消耗token数为85
        # input_tokens = text_tokens + image_tokens
        # print(f"输入token数{input_tokens}")  # 固定为 80+85 = 165
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                # {"role": "system", "content": "你是一个熟悉 ECharts 的前端开发专家"},
                {"role": "user", "content": text_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(compressed_img_data).decode()}}]}
            ],
        )

        response_text = response.choices[0].message.content

         # 计算输出token
        output_tokens = self.count_tokens(response_text)
        # print(f"输出token数{output_tokens}")
        self.token_count += output_tokens
        # print("收到的结果为：" + response_text)
        start = response_text.find("{")
        end = response_text.rfind("}")
        json_text = response_text[start:end+1]

        return json_text
    

     # 获取缺少代码的数据的数量
    @staticmethod
    def get_record_count(score_file):

        df = pd.read_csv(score_file)
        
        # 筛选出 有code数据 的记录中 没有图像数据 的记录
        condition = (~df['image'].isna()) & (df['code'].isna())
        filtered_df = df[condition]

        # 获取数量
        count = len(filtered_df)

        return count

    @staticmethod
    def check_file_existence(file, file_array):
        return file in file_array
