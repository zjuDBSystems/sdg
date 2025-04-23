
import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor

from ...event import global_message_queue, EventType, EventResponse

from playwright.sync_api import sync_playwright
def test_renderability(js_code_path: str, screenshot_folder: str) -> bool:
    try:
        # print('test file: ' + js_code_path)
        global_message_queue.put(EventResponse(EventType.REASONING, f'渲染测试 {js_code_path}...'))
        # 从文件中读取JavaScript代码
        with open(js_code_path, "r", encoding="utf-8") as f:
            js_code = f.read()
        # print(js_code_path)
        if js_code.strip().startswith("{") and js_code.strip().endswith("}"):
            js_code = "option = " + js_code

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page()


            page.set_content('<div id="main" style="width: 600px;height:400px;"></div>')  # 设置容器

            # 在页面中引入 ECharts 和用户的代码
            page.add_script_tag(
                url='https://cdn.jsdelivr.net/npm/echarts@5.1.0/dist/echarts.min.js')  # 加载 ECharts 库
            page.add_script_tag(content=js_code)  # 加载用户的 ECharts 配置代码
            # page.set_content(html_content)
            initialization_code = """
                            var chart = echarts.init(document.getElementById('main'));
                            chart.setOption(option);
                            """
            # print('waiting rendering: ' + js_code_path)
            global_message_queue.put(EventResponse(EventType.REASONING, f'等待渲染完成 {js_code_path}...'))
            page.evaluate(initialization_code)
                    # 等待图表渲染完成
            page.wait_for_selector('#main canvas')  # 等待Canvas渲染

            page.wait_for_function('document.querySelector("#main canvas").clientWidth > 0'
                                           )  # 等待Canvas有宽度
            sleep(0.7)
                    # 截图保存到指定文件夹，截图名与代码文件名相同
            screenshot_name = os.path.basename(js_code_path).replace('.json', '.png')
            screenshot_path = os.path.join(screenshot_folder, screenshot_name)
            page.screenshot(path=screenshot_path)

                    # 检查渲染是否成功（检查页面内容）
            content = page.evaluate('document.getElementById("main").innerHTML')
            global_message_queue.put(EventResponse(EventType.REASONING, f'渲染完成 {js_code_path}...'))
        if content:

            # browser.close()
            return True  # 渲染成功
        else:

            # browser.close()
            return False  # 渲染失败

    except Exception as e:
        print(f"Execution error: {e}")
        return False

def process_js_folder(js_folder: str, screenshot_folder: str) :
    """遍历代码文件夹并评判每个文件的渲染结果"""
    # 获取文件夹中的所有JS文件
    js_files = [f for f in os.listdir(js_folder) if f.endswith(".json")]

    total_files = len(js_files)
    if total_files == 0:
        print("该文件夹没有JS文件.")
        return 0.0

    # 创建保存截图的文件夹
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    file_scores = {}
    passed_count = 0

    # for js_file in js_files:
    #     js_path = os.path.join(js_folder, js_file)
    #     success = test_renderability(js_path, screenshot_folder)
    #     score = 100 if success else 0
    #     file_scores[js_file] = score
    #     passed_count += 1 if success else 0
    with ThreadPoolExecutor() as executor:
        futures = []
        for js_file in js_files:
            js_path = os.path.join(js_folder, js_file)
            future = executor.submit(test_renderability, js_path, screenshot_folder)
            futures.append(future)

        for i, future in enumerate(futures):
            js_file = js_files[i]
            success = future.result()
            score = 100 if success else 0
            file_scores[js_file] = score
            passed_count += 1 if success else 0


    # 计算可渲染通过率
    renderability_score = (passed_count / total_files) * 100
    return renderability_score,file_scores


def evaluate_renderability(js_folder: str, screenshot_folder: str)->float:
    return process_js_folder(js_folder, screenshot_folder)
