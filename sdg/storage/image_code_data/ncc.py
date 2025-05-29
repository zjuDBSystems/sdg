import os
import cv2
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import numpy as np
from time import sleep
import json
from concurrent.futures import ThreadPoolExecutor
from playwright.sync_api import sync_playwright


# def calculate_ssim(image1_path, image2_path):
#     # image1_path = image1_path.encode('utf-8').decode('gbk')
#     # image2_path = image2_path.encode('utf-8').decode('gbk')
#     image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
#     image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
#     cv2.imwrite("./处理前_image1.png", image1)
#     cv2.imwrite("./处理前_image2.png", image2)
#
#     if image1.shape != image2.shape:
#         image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
#
#         # 创建掩码：认为白色区域为背景
#     _, mask1 = cv2.threshold(image1, 240, 255, cv2.THRESH_BINARY)
#     _, mask2 = cv2.threshold(image2, 240, 255, cv2.THRESH_BINARY)
#
#         # 使用掩码去除空白区域
#     image1 = cv2.bitwise_and(image1, image1, mask=mask1)
#     image2 = cv2.bitwise_and(image2, image2, mask=mask2)
#
#     cv2.imwrite("./image1.png", image1)
#     cv2.imwrite("./image2.png", image2)
#     ssim_value, _ = ssim(image1, image2, full=True)
#     # print(ssim_value)
#     return (int)(ssim_value*100)


def generate_screenshots(js_dir, screenshot_folder):
    """生成所有JSON配置的截图"""

    def render_js(js_path, screenshot_folder):
        try:
            with open(js_path, "r", encoding="utf-8") as f:
                js_code = f.read()
            
            js_code_dict = json.loads(js_code)
            js_code_dict['animation'] = False  # 禁用动画
            js_code = json.dumps(js_code_dict, ensure_ascii=False)

            if js_code.strip().startswith("{") and js_code.strip().endswith("}"):
                js_code = "option = " + js_code

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_content('<div id="main" style="width:600px;height:400px;"></div>')
                page.add_script_tag(url='https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js')
                page.add_script_tag(content=js_code)
                page.evaluate("""
                    var chart = echarts.init(document.getElementById('main'));
                    chart.setOption(option);
                """)
                page.wait_for_selector('#main canvas', timeout=5000)
                screenshot_name = os.path.basename(js_path).replace('.json', '.png')
                screenshot_path = os.path.join(screenshot_folder, screenshot_name)
                # page.screenshot(path=screenshot_path)
                chart_div = page.locator('#main')
                chart_div.screenshot(
                    path=screenshot_path
                )
                browser.close()
                # print(screenshot_name,"生成成功")
                return True
        except Exception as e:
            # print(f"渲染失败: {os.path.basename(js_path)} - {str(e)}")
            return False

    # 创建输出目录
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    # 多线程处理渲染任务
    js_files = [os.path.join(js_dir, f) for f in os.listdir(js_dir) if f.endswith('.json')]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(render_js, js_path, screenshot_folder) for js_path in js_files]
        [future.result() for future in futures]


def calculate_ncc(image1_path, image2_path):
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    # err /= float(image1.shape[0] * image1.shape[1])
    return (int)(max(0, np.sum((image1 - np.mean(image1)) * (image2 - np.mean(image2))) / (
            np.std(image1) * np.std(image2) * image1.size) * 100))
    # return


def build_code_mapping(csv_path):
    """构建代码文件到原始图像的映射字典"""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['code', 'image'])
    return dict(zip(df['code'], df['image']))


def process_screenshots(screenshot_folder, image_folder, code_map):
    """处理所有渲染截图并计算SSIM"""
    score_dict = {}
    # print(screenshot_folder)
    # print(image_folder)
    for screenshot_name in os.listdir(screenshot_folder):
        # 获取对应的代码文件名
        code_file = os.path.splitext(screenshot_name)[0] + '.json'

        # 查找原始图像文件名
        original_image = code_map.get(code_file)
        # print(original_image)
        if not original_image:
            continue

        # 构建完整路径
        screenshot_path = os.path.join(screenshot_folder, screenshot_name)
        original_path = os.path.join(image_folder, original_image)
        # print(screenshot_path)
        # print(original_path)

        # 计算SSIM
        if os.path.exists(original_path):
            similarity = calculate_ncc(original_path, screenshot_path)
            score_dict[original_image] = similarity  # 使用原始图像文件名作为键
            # print(f"已处理: {original_image} → {similarity}%")

    return score_dict


# def evaluate_ssim(csv_path, image_folder, screenshot_folder):
#     """主逻辑"""
#     code_map = build_code_mapping(csv_path)
#     print(code_map)
#     score_dict = process_screenshots(screenshot_folder, image_folder, code_map)
#
#     if score_dict:
#         avg_score = sum(score_dict.values()) / len(score_dict)
#         # print(f"\n有效样本数: {len(score_dict)}")
#         # print(f"平均SSIM相似度: {avg_score:.1f}%")
#         return avg_score, score_dict
#     else:
#         print("未找到有效数据对")
#         return 0.0, {}

def evaluate_ncc(csv_path, image_folder, screenshot_folder, js_dir):
    """整合后的主函数"""
    # Step 1: 生成截图
    generate_screenshots(js_dir, screenshot_folder)

    # Step 2: 计算相似度
    code_map = build_code_mapping(csv_path)
    score_dict = process_screenshots(screenshot_folder, image_folder, code_map)

    print("========== 渲染截图与图像的匹配度指标评估结果 ==========")
    if score_dict:
        avg_score = sum(score_dict.values()) / len(score_dict)
        print(f"\n所有样本的渲染截图与原始图像的平均匹配度得分: {avg_score:.2f}%")

        low_score_pairs = [(image, score) for image, score in score_dict.items() if score < 70]  # 设定阈值为70，可调整
        if avg_score >= 90:
            print("整体上，渲染截图与原始图像高度匹配，说明渲染过程非常准确，配置和渲染环境等方面表现极佳，无需任何调整。")
        elif avg_score > 70:
            print("整体的匹配度处于较好水平，大部分渲染截图与原始图像相似度较高，但仍有部分截图存在一定偏差。这可能是由于配置细节差异、渲染环境的微小变化或数据本身的特点导致的，建议对匹配度较低的样本进行检查和分析。")
            if low_score_pairs:
                print("以下是匹配度得分低于70分的数据对及其分数：")
                for image, score in low_score_pairs:
                    print(f"原始图像: {image}, 分数: {score}")
        else:
            print(
                "整体的匹配度较低，渲染截图与原始图像之间存在明显差异。这可能是因为配置错误、渲染环境不稳定、数据格式不兼容等多种原因造成的。需要全面检查配置文件、渲染代码以及数据本身，找出问题根源并加以解决，以提高渲染截图与原始图像的匹配度。")
            if low_score_pairs:
                print("以下是匹配度得分低于70分的数据对及其分数：")
                for image, score in low_score_pairs:
                    print(f"原始图像: {image}, 分数: {score}")

        return avg_score, score_dict
    else:
        print("未找到有效数据对，无法计算匹配度得分，请检查输入的CSV文件、图像文件夹和截图文件夹的内容及配置是否正确。")
        return 0.0, {}