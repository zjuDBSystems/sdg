
import pandas as pd
import os
from playwright.sync_api import sync_playwright
import json
from typing import override

from .operator import Operator, Field, Meta
from ..storage.dataset import DataType
from ..task.task_type import TaskType
from concurrent.futures import ThreadPoolExecutor
import threading

class EchartsToImageOperator(Operator):
    """EchartsToImageOperator is an operator that generates images from echarts code.
    """
    @classmethod
    @override
    def accept(cls, data_type, task_type) -> bool:
        if data_type == DataType.CODE and task_type == TaskType.AUGMENTATION:
            return True
        return False

    @classmethod
    @override
    def get_config(cls) -> list[Field]:

        return []
    
    @classmethod
    @override
    def get_meta(cls) -> Meta:
        return Meta(
            name='EchartsToImageOperator',
            description='Generates images from echarts code.'
        )

    @override
    def execute(self, dataset) -> None:
        df = pd.read_csv(dataset.meta_path)
        code_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.CODE][0]
        img_dir = [dir for dir in dataset.dirs if dir.data_type == DataType.IMAGE][0]
        code_files = df[DataType.CODE.value].tolist()
        img_files = df[DataType.IMAGE.value].tolist()

        poc_code_files = []

        for index, code_file_name in enumerate(code_files):

            if pd.isna(code_file_name):
                continue
            
            img_file_name = os.path.splitext(code_file_name)[0]+'.png'

            if not self.check_file_existence(img_file_name, img_files):

                poc_code_files.append((index, code_file_name))

        success_files = self.generate_imgs(code_dir.data_path, poc_code_files, img_dir.data_path)

        # modify csv file
        for file_index, img_file_name in success_files:
            df.at[file_index, DataType.IMAGE.value] = img_file_name

        # save the modified csv
        df.to_csv(dataset.meta_path, index=False)
        print(f"csv文件已更新，保存至{dataset.meta_path}")

    @staticmethod
    def generate_imgs(code_dir, code_files, img_dir):
        # 初始化线程安全的结果容器
        results = []
        results_lock = threading.Lock()

        # 记录生成成功与失败的数量
        success_count= 0

        def gen_single(index, code_path, img_dir):
            nonlocal success_count
            try:
                with open(code_path, "r", encoding='utf-8') as f:
                    js_code = f.read()
                
                js_code_dict = json.loads(js_code)
                js_code_dict['animation'] = False  # 禁用动画
                js_code = json.dumps(js_code_dict, ensure_ascii=False)

                if js_code.strip().startswith("{") and js_code.strip().endswith("}"):
                    js_code = "option = " + js_code

                with sync_playwright() as p:
                    browser = p.chromium.launch(
                        headless=True,
                        args=[
                            '--disable-web-security',  # 禁用同源策略
                            '--ignore-certificate-errors'  # 忽略证书错误
                        ]
                    )
                    page = browser.new_page()

                
                    # 加载页面内容
                    page.set_content('<div id="main" style="width:600px;height:400px;"></div>')
                    page.add_script_tag(url='https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js')
                    page.add_script_tag(content=js_code)
                    page.evaluate("""
                        var chart = echarts.init(document.getElementById('main'));
                        chart.setOption(option);
                    """)
                    
                    page.wait_for_selector('#main canvas', timeout=5000)  # 等待Canvas渲染

                    # 截图配置
                    chart_div = page.locator('#main')
                    # 存储到指定路径
                    img_name = os.path.basename(code_path).replace('.json', '.png')
                    img_path = os.path.join(img_dir, img_name)
                    chart_div.screenshot(
                        type='png',
                        path = img_path
                    )
                    browser.close()
                    # 线程安全地保存结果
                    with results_lock:
                        results.append((index, img_name))
                        success_count = success_count+1
                    return True

            except Exception as e:
                return False
        
        # 创建输出目录
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        # 多线程处理渲染任务
        poc_code_files = []
        for code_index, f in code_files:
            poc_code_files.append((code_index, os.path.join(code_dir, f)))

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(gen_single, code_index, code_path, img_dir) for code_index, code_path in poc_code_files]
            [future.result() for future in futures]

        print(f"共尝试生成{len(poc_code_files)}张图片，{success_count}张成功，{len(poc_code_files)-success_count}张失败")
        # 返回处理成功的图像信息
        return results


    @staticmethod
    def check_file_existence(file, file_array):
        return file in file_array