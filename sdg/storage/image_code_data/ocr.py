import os
import pandas as pd
from openai import OpenAI
import base64


# 构造 client
client = OpenAI(
    api_key="",#混元大模型api
    base_url="https://api.hunyuan.cloud.tencent.com/v1",
)


# 从JS代码文件中读取内容
def load_js_code(js_code_path):
    try:
        with open(js_code_path, 'r', encoding='utf-8') as f:
            js_code = f.read()
        return js_code
    except Exception as e:
        print(f"Error reading or processing the file: {e}")
        return None


# 使用混元大模型进行匹配度打分
def evaluate_with_hunyuan(image_path, js_code):
    if js_code is None:
        return 0
    image_url = f"{os.path.abspath(image_path)}" if os.path.exists(image_path) else ""
    encoded_image_bs64=""
    with open(image_url, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read())
        encoded_image_bs64="data:image/jpeg;base64," + encoded_image.decode('utf-8')
        # print("data:image/jpeg;base64," + encoded_image.decode('utf-8'))
    # print(encoded_image_bs64)
    prompt = f"<js代码>\n{js_code}\n请你判断这个代码和这个图像是否匹配，只需考虑是否匹配，无需考虑其他方面，即假如将这份代码渲染，是否会是这张图的样子，给出分数，得分范围是0-100，你的输出应只包含数字形式的得分，无需理由等其他任何东西"
    # print(prompt)
    try:
        completion = client.chat.completions.create(
            model="hunyuan-vision",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": encoded_image_bs64
                            }
                        }
                    ]
                },
            ],
        )
        score = completion.choices[0].message.content.strip()
        # print(score)
        try:
            score = float(score)
            if 0 <= score <= 100:
                return score
            else:
                print(f"Invalid score: {score}, should be between 0 and 100.")
                return 0
        except ValueError:
            print(f"Could not convert score to float: {score}")
            return 0
    except Exception as e:
        print(f"Error getting score from Hunyuan: {e}")
        return 0


# 总体功能函数
def evaluate_ocr(csv_path, image_folder, code_folder):
    """返回 (平均分, {代码文件: 得分}) """
    df = pd.read_csv(csv_path)
    code_scores = {}
    total_score = 0.0
    valid_pairs = 0

    for index, row in df.iterrows():
        code_file = row['code_file']
        try:
            # 获取完整路径
            image_path = os.path.join(image_folder, row['image_file'])
            code_path = os.path.join(code_folder, code_file)

            # 读取代码内容
            js_code = load_js_code(code_path)
            if js_code is None:
                code_scores[code_file] = 0.0
                continue

            # 获取评估得分
            score = evaluate_with_hunyuan(image_path, js_code)

            # 记录得分
            code_scores[code_file] = float(score)
            total_score += score
            valid_pairs += 1

            # print(f"已处理 {code_file}: {score:.1f} 分")

        except Exception as e:
            print(f"处理失败 {code_file}: {str(e)}")
            code_scores[code_file] = 0.0

    # 计算平均分
    avg_score = total_score / valid_pairs if valid_pairs > 0 else 0.0
    return avg_score, code_scores





