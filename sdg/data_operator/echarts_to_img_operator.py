
import pandas as pd
import os
from playwright.sync_api import sync_playwright  

from typing import override
from .operator import Operator, Field, Meta
from ..storage.dataset import DataType
from ..task.task_type import TaskType
from ..event import global_message_queue, EventType, EventResponse

class EchartsToImageOperator(Operator):
    """EchartsToImageOperator is an operator that generates images from echarts code.
    """
    # def __init__(self, **kwargs)

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

        for index, code_file_name in enumerate(code_files):

            if pd.isna(code_file_name):
                continue
            
            img_file_name = os.path.splitext(code_file_name)[0]+'.png'

            if not self.check_file_existence(img_file_name, img_files):

                code_file_path = os.path.join(code_dir.data_path,code_file_name)
                with open(code_file_path, 'rb') as f:
                    code = f.read().decode('utf-8')
            
                img_file_path = os.path.join(img_dir.data_path, img_file_name)

                with open(img_file_path, 'wb') as f:
                    bytes = self.generate_echarts_png(code)
                    if bytes is None:
                        continue
                    f.write(bytes)

                # modify csv file
                df.at[index, DataType.IMAGE.value] = img_file_name

        # save the modified csv
        df.to_csv(dataset.meta_path, index=False)

    @staticmethod
    def generate_echarts_png(echarts_option):
        start = echarts_option.find("{")
        end = echarts_option.rfind("}")
        echarts_option_json = echarts_option[start:end+1]

        # 设置页面内容
        html_content = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <script src="https://cdn.bootcdn.net/ajax/libs/echarts/5.4.3/echarts.min.js"></script>
        </head>
        <body>
            <div id="chart" style="width:1200px;height:800px;"></div>
            <script>
                // Playwright适配方案
                document.addEventListener('DOMContentLoaded', () => {{
                    const chart = echarts.init(document.getElementById('chart'));
                    chart.setOption({echarts_option_json});
                    chart.on('rendered', () => {{
                        document.title = 'RENDER_DONE';  // 保持标题标记
                    }});
                }});
            </script>
        </body>
        </html>
        '''
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=[
                    '--disable-web-security',  # 禁用同源策略
                    '--ignore-certificate-errors'  # 忽略证书错误
                ]
            )
            page = browser.new_page()

            try:
                # 加载页面内容
                page.set_content(html_content)
                
                # 双重等待策略
                page.wait_for_selector('#chart canvas', timeout=60_000)  # 等待Canvas渲染
                page.wait_for_function('''() => {
                    return document.title === 'RENDER_DONE' &&
                        document.querySelector('#chart canvas').clientWidth > 0
                }''', timeout=10_000)  # 等待渲染标记
                
                # 先检测canvas元素是否存在
                page.wait_for_selector('#chart canvas', state='attached', timeout=10_000)
                # 再检测渲染完成标志
                page.wait_for_function('document.title === "RENDER_DONE"', timeout=10_000)
                # 最后添加2秒保险延迟（针对复杂渲染场景）
                page.wait_for_timeout(2000)  # 等同于 time.sleep(2)

                # 截图配置
                chart_div = page.locator('#chart')
                # 存储到指定路径
                # chart_div.screenshot(
                #     path='api_result/tongyi.jpg',
                #     type='jpeg',
                #     quality=95
                # )
                screenshot_bytes = chart_div.screenshot(
                type='png',
                # quality=95
                )
                browser.close()
                global_message_queue.put(EventResponse(event=EventType.REASONING, data="✅ 图片数据制备成功"))
                print("✅ 成功生成 png")
                return screenshot_bytes
            except Exception as e:
                global_message_queue.put(EventResponse(event=EventType.REASONING, data="❌ 图片数据制备失败"))
                print(f"❌ 生成失败: {str(e)}")
                # 生成调试文件
                with open('error_debug.html', 'w') as f:
                    f.write(html_content)
                print("已生成 error_debug.html 供手动调试")
                
            finally:
                browser.close()

    @staticmethod
    def check_file_existence(file, file_array):
        return file in file_array