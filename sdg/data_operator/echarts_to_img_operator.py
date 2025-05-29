
import pandas as pd
import os
from playwright.sync_api import sync_playwright
import json

from typing import override
from .operator import Operator, Field, Meta
from ..storage.dataset import DataType
from ..task.task_type import TaskType
from ..event import global_message_queue, EventType, EventResponse
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import threading

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

        poc_code_files = []

        for index, code_file_name in enumerate(code_files):

            if pd.isna(code_file_name):
                continue
            
            img_file_name = os.path.splitext(code_file_name)[0]+'.png'

            if not self.check_file_existence(img_file_name, img_files):

                poc_code_files.append((index, code_file_name))

                # code_file_path = os.path.join(code_dir.data_path,code_file_name)
                # with open(code_file_path, 'rb') as f:
                #     code = f.read().decode('utf-8')
            
                # img_file_path = os.path.join(img_dir.data_path, img_file_name)

                # bytes = self.generate_echarts_png(code)
                # if bytes is None:
                #     continue
                # with open(img_file_path, 'wb') as f:
                #     f.write(bytes)
                #     print(f"已保存至{img_file_path}")

                # # modify csv file
                # df.at[index, DataType.IMAGE.value] = img_file_name

        # for d, f in poc_code_files:
        #     print(f"code_index = {d}, f = {f}" )

        success_files = self.generate_img_parallel(code_dir.data_path, poc_code_files, img_dir.data_path)

        # modify csv file
        for file_index, img_file_name in success_files:
            df.at[file_index, DataType.IMAGE.value] = img_file_name

        # save the modified csv
        df.to_csv(dataset.meta_path, index=False)
        print(f"csv文件已更新，保存至{dataset.meta_path}")

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
                # page.wait_for_timeout(2000)  # 等同于 time.sleep(2)

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
    def generate_img_parallel(code_dir, code_files, img_dir):

        # 初始化线程安全的结果容器
        results = []
        results_lock = threading.Lock()

        # 记录生成成功与失败的数量
        success_count= 0

        def gen_single(index, code_path, img_dir):
            try:
                with open(code_path, "r", encoding="utf-8") as f:
                    code_content = f.read()

                if code_content.strip().startswith("{") and code_content.strip().endswith("}"):
                    code_content_dict = json.loads(code_content)
                    code_content_dict['animation'] = False  # 禁用动画
                    code_content = json.dumps(code_content_dict, ensure_ascii=False)
                    code_content = "option = " + code_content
                
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.set_content('<div id="main" style="width:600px;height:400px;"></div>')
                    page.add_script_tag(url='https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js')
                    page.add_script_tag(content=code_content)
                    page.evaluate("""
                        var chart = echarts.init(document.getElementById('main'));
                        chart.setOption(option);
                    """)
                    page.wait_for_selector('#main canvas', timeout=5000)
                    img_name = os.path.basename(code_path).replace('.json', '.png')
                    img_path = os.path.join(img_dir, img_name)
                    # page.screenshot(path=img_path)
                    chart_div = page.locator('#main')
                    chart_div.screenshot(
                        path=img_path
                    )
                    browser.close()

                    # 线程安全地保存结果
                    with results_lock:
                        results.append((index, img_name))

                    print(f"{img_name}生成成功")

                    return True
            except Exception as e:
                # print(f"渲染失败: {os.path.basename(js_path)} - {str(e)}")
                print(f"{code_path}对应图像生成失败")
                return False

        def gen_single_copy(index, code_path, img_dir):
            with open(code_path, "r", encoding='utf-8') as f:
                code_content = f.read()
            start = code_content.find("{")
            end = code_content.rfind("}")
            echarts_option_json = code_content[start:end+1]
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
                    # page.wait_for_timeout(2000)  # 等同于 time.sleep(2)

                    # 截图配置
                    chart_div = page.locator('#chart')
                    # 存储到指定路径
                    img_name = os.path.basename(code_path).replace('.json', '.png')
                    img_path = os.path.join(img_dir, img_name)
                    chart_div.screenshot(
                        type='png',
                        path = img_path
                    )
                    browser.close()
                    # global_message_queue.put(EventResponse(event=EventType.REASONING, data="✅ 图片数据制备成功"))
                    # print(f"✅ {img_name}成功生成")
                    # 线程安全地保存结果
                    with results_lock:
                        results.append((index, img_name))
                        success_count = success_count+1
                    return True
                except Exception as e:
                    # global_message_queue.put(EventResponse(event=EventType.REASONING, data="❌ 图片数据制备失败"))
                    # print(f"❌ {code_path}对应图片生成失败")
                    return False
                    # 生成调试文件
                    # with open('error_debug.html', 'w') as f:
                    #     f.write(html_content)
                    # print("已生成 error_debug.html 供手动调试")
                finally:
                    browser.close()

            
        # 创建输出目录
        if not os.path.exists(img_dir):
            print("输出图像所存储的目录不存在，已为其创建目录")
            os.makedirs(img_dir)

        # 多线程处理渲染任务
        # poc_code_files = [{code_index, os.path.join(code_dir, f)} for code_index, f in code_files]
        poc_code_files = []
        for code_index, f in code_files:
            # print(f"code_index = {code_index}, f = {f}" )
            poc_code_files.append((code_index, os.path.join(code_dir, f)))

        with ThreadPoolExecutor() as executor:
            # futures = [executor.submit(gen_single, code_index, code_path, img_dir) for code_index, code_path in poc_code_files]
            futures = [executor.submit(gen_single_copy, code_index, code_path, img_dir) for code_index, code_path in poc_code_files]
            [future.result() for future in futures]

        print(f"共尝试生成{len(poc_code_files)}张图片，{success_count}张成功，{len(poc_code_files)-success_count}张失败")
        # 返回处理成功的图像信息
        return results



    @staticmethod
    def check_file_existence(file, file_array):
        return file in file_array