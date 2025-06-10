import random
from PIL import Image, ImageDraw, ImageFont
from typing import override, Dict
import io
import pandas as pd
import os
import shutil
from tqdm import tqdm

from .operator import Meta, Operator, Field
from ..storage.dataset import DataType
from ..task.task_type import TaskType

class ImageRobustnessEnhancer(Operator):
    """ImageRobustnessEnhancer is an operator that enhances the robustness of images
    by adding random interference elements and text descriptions.

    Attributes:
        add_watermark: Whether to add a semi-transparent watermark.
        add_noise: Whether to add noise to the image.
        add_text: Whether to add text descriptions around the image.
    """

    def __init__(self, **kwargs):
        self.add_watermark: bool = kwargs.get('add_watermark', True)
        self.watermark_count: int = kwargs.get('water_count', 25)
        self.add_noise: bool = kwargs.get('add_noise', True)
        self.add_text: bool = kwargs.get('add_text', True)
        self.text_count: int = kwargs.get('text_count', 15)
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
            Field('add_watermark', Field.FieldType.BOOL,
                  'Add a semi-transparent watermark to the image.', True),
            Field('watermark_count', Field.FieldType.NUMBER,
                  'The number of watermark.', 25),
            Field('add_noise', Field.FieldType.BOOL,
                  'Add noise to the image.', True),
            Field('add_text', Field.FieldType.BOOL,
                  'Add text descriptions around the image.', True),
            Field('text_count', Field.FieldType.NUMBER,
                  'The number of text.', 15)
        ]

    @classmethod
    @override
    def get_meta(cls) -> Meta:
        return Meta(
            name='ImageRobustnessEnhancer',
            description='Enhances the robustness of images by adding random interference elements and text descriptions.'
        )

    def get_cost(self, dataset) -> Dict:
        cost = {}
        # operator name
        cost["name"] = "ImageRobustnessEnhancer"
        # records count
        cost["ri"] = self.get_pending_files(dataset.meta_path, "image")
        # time of one record
        cost["ti"] = 0.096
        # cpi time of one record
        cost["ci"] = 0.095
        # operator type
        cost["type"] = "CPU"
        return cost

    @override
    def execute(self, dataset) -> None:

        df = pd.read_csv(dataset.meta_path, na_values=['nan', 'None', ''])
        img_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.IMAGE][0]
        img_files = df[DataType.IMAGE.value].tolist()
        code_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.CODE][0]
        code_files = df[DataType.CODE.value].tolist()
        type_name = df["type"].tolist()

        # self.get_cost(dataset)

        for index, img_name in enumerate(tqdm(img_files, desc="扰动进度")):
            if pd.isna(img_name):
                continue
            # print(img_name)
            file_path = os.path.join(img_dir.data_path, img_name)
            with open(file_path, 'rb') as f:
                img_data = f.read()
            # 将字节数据转换为 BytesIO 对象
            img_byte_arr = io.BytesIO(img_data)
            with Image.open(img_byte_arr) as img:
                if self.add_watermark:
                    img = self._add_watermark(img)
                if self.add_noise:
                    img = self._add_noise(img)
                if self.add_text:
                    img = self._add_text(img)

                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

            # new image
            new_img_path = os.path.join(img_dir.data_path, "r_"+img_name)
            with open(new_img_path, 'wb') as f:
                f.write(img_byte_arr)
                # print(f"扰动后的图像已保存至{new_img_path}")

            # old code
            code_name = code_files[index]
            # print(code_name)
            # if pd.isna(code_name):
            #     code_name=""
            #     print("该扰动图像没有原代码文件")
            # else:
            #     old_code_path = os.path.join(code_dir.data_path, code_name)
            #     copy_old_code_path = os.path.join(code_dir.data_path, "r_"+code_name)
            #     shutil.copy2(old_code_path, copy_old_code_path)
            #     code_name = "r_"+code_name
            #     print(f"该扰动图像的原代码文件已复制到{copy_old_code_path}")


            new_data = pd.DataFrame({"image": ["r_"+img_name], "code": [code_name], "type": [type_name[index]]})
            df = pd.concat([df, new_data], ignore_index=True)  # 合并数据
        
        # 保存新数据
        df.to_csv(dataset.meta_path, index=False)


    def _add_watermark(self, img):
        watermark = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(watermark)
        font = ImageFont.load_default()
        # print(f"当前使用的字体: {font}")  
        text = "Sample Watermark"

        # 水印数量
        num_watermarks = self.watermark_count  
        # 随机生成颜色，这里固定了透明度是128（0是完全透明，255是完全不透明）
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = (r, g, b, 128)
        

        for _ in range(num_watermarks):
            # 使用 textbbox 方法计算文本的边界框
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            text_width = right - left
            text_height = bottom - top
            
            x = random.randint(0, img.width - text_width)
            y = random.randint(0, img.height - text_height)

            draw.text((x, y), text, font=font, fill=color)
       
        watermarked = Image.alpha_composite(img.convert('RGBA'), watermark)
        return watermarked.convert('RGB')

    def _add_noise(self, img):
        # 生成与原始图片尺寸和模式相同的噪声图片
        noise = Image.effect_noise(img.size, 100).convert(img.mode)
        noisy_img = Image.blend(img, noise, 0.1)
        return noisy_img

    def _add_text(self, img):
        texts = ["This is a related description.", "Another description here."]
        text = random.choice(texts)
        new_width = img.width + 200
        new_height = img.height + 200
        new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        new_img.paste(img, (100, 100))
        draw = ImageDraw.Draw(new_img)
        font = ImageFont.load_default()
        
        # 文本数量
        num_text = self.text_count

        for _ in range(num_text):
            # # 使用 textbbox 方法计算文本的边界框
            # left, top, right, bottom = draw.textbbox((0,0), text, font=font)
            # text_width = right - left
            # text_height = bottom - top
        
            # 随机选择文本位置
            # x = random.randint(0, new_width - text_width)
            # y = random.randint(0, new_height - text_height)

            x = random.randint(0, new_width - 100)
            y1 = random.randint(0, 85)
            y2 = random.randint(new_height-85, new_height)
            y = random.choice([y1, y2])
            # 随机生成颜色
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color = (r, g, b)

            draw.text((x, y), text, font=font, fill=color)

        return new_img

    # def _add_text(self, img):
    #     texts = ["This is a related description.", "Another description here."]
    #     text = random.choice(texts)
    #     new_width = img.width + 200
    #     new_height = img.height + 200
    #     new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    #     new_img.paste(img, (100, 100))
    #     draw = ImageDraw.Draw(new_img)
    #     font = ImageFont.load_default()
    #     x = random.randint(0, new_width - 100)
    #     y = random.randint(0, 50)
    #     draw.text((x, y), text, font=font, fill=(0, 0, 0))
    #     return new_img

    @staticmethod
    def get_pending_files(csv_path, file_type):
        # 读取 CSV 文件
        df = pd.read_csv(csv_path)

        # 统计image字段的不同值数量
        column_name = file_type 
        value_counts = df[column_name].nunique()

        print(f"待处理的image数量为{value_counts}")
        return value_counts
