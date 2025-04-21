from openai import OpenAI
import json, re
import torch
import torch.nn as nn
import torch.nn.init as init

# Step 1: Deepseek接口返回结果
def get_llm_analysis(client, code_quality_analysis, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # 构造提示词
            prompt = f"""### **任务描述**
你将收到一份代码数据集的质量评估报告。该报告包含 **11个质量评估维度**，每个维度都有一个分数（范围为 1-100），用于衡量数据集在该维度上的质量表现。所有维度的分数越高，表示该维度的质量越好。

你的任务是根据这些维度的分数，分析它们对整体数据质量的影响程度，并返回一个 Python 列表。列表中应包括这 11 个维度，且维度的排序需要反映其对整体数据质量的重要性：越靠前的维度，其对整体质量的影响越大。

这里我们定义，一个维度对整体质量的影响受到维度本身和该维度质量分数的影响，质量分数越低约影响整体质量，但需要同时考虑维度本身，可能有的维度哪怕分数低，影响也不是特别大，而有的维度哪怕分数没那么低，影响也很大。

---

### **输入说明**
输入是一个包含 11 个维度及其对应分数的字典，格式如下：

```python
{{
"语法检测": 0.85,
"可渲染性检测": 0.92,
"配置项完整检测": 0.78,
"图像与渲染截图的SSIM": 0.88,
"图像OCR检测的文字与配置项的余弦相似度": 0.81,
"图表类型均衡性": 0.75,
"配置项多样性": 0.93,
"代码重复": 0.68,
"图像重复": 0.72,
"联合重复": 0.80,
"数据量": 0.89
}}
```

---

### **任务要求**
1. **分析维度重要性**：
- 根据每个维度的分数，结合其对整体数据质量的影响程度，判断哪些维度更重要。
- 分数高且对数据质量影响大的维度应排在前面。

2. **输出格式**：
- 返回一个 Python 列表，列表中包含 11 个维度名称，按重要性从高到低排序。
- 输出必须严格遵循以下格式：

```python
["维度1", "维度2", "维度3", ..., "维度11"]
```

---

### **示例**
假设输入如下：

```python
{{
"语法检测": 0.85,
"可渲染性检测": 0.92,
"配置项完整检测": 0.78,
"图像与渲染截图的SSIM": 0.88,
"图像OCR检测的文字与配置项的余弦相似度": 0.81,
"图表类型均衡性": 0.75,
"配置项多样性": 0.93,
"代码重复": 0.68,
"图像重复": 0.72,
"联合重复": 0.80,
"数据量": 0.89
}}
```

经过分析后，输出应为：

```python
["配置项多样性", "可渲染性检测", "数据量", "图像与渲染截图的SSIM", "语法检测", "图像OCR检测的文字与配置项的余弦相似度", "联合重复", "配置项完整检测", "图像重复", "图表类型均衡性", "代码重复"]
```

### **你的任务**
{json.dumps(code_quality_analysis, indent=4, ensure_ascii=False)}
"""
            # 调用 API 获取响应
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )


            # 提取模型返回的内容
            order = response.choices[0].message.content

            print(order)

            # 使用正则表达式提取 JSON 格式内容
            pattern = r'```python(.*?)```'
            match = re.findall(pattern, order, re.DOTALL)
            if match:
                # 解析 JSON 数据
                return json.loads(match[0].strip())
            else:
                raise ValueError("Response does not contain valid JSON block.")

        except ValueError as ve:
            # 处理 JSON 解析错误
            print(f"ValueError: {ve}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries})")


        except Exception as e:
            # 处理其他异常（如 API 错误、网络问题等）
            print(f"An unexpected error occurred: {e}")
            retries += 1
            print(f"Retrying... ({retries}/{max_retries})")


    # 如果达到最大重试次数，抛出异常
    raise RuntimeError("Failed to process the request after multiple attempts.")

# Step 2: 构建MLP模型（预训练权重假设已存在）
class MLP(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=20, output_dim=11):
        """
        初始化一个简单的多层感知机（MLP）模型。

        参数:
            input_dim (int): 输入维度，默认为 11。
            hidden_dim (int): 隐藏层维度，默认为 20。
            output_dim (int): 输出维度，默认为 11。
        """
        super(MLP, self).__init__()

        # 定义网络层
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 隐藏层到输出层

        # 随机初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """
        随机初始化网络的权重和偏置。
        """
        for layer in [self.fc1, self.fc2]:
            # 权重初始化为小的随机值（正态分布）
            init.normal_(layer.weight, mean=0, std=0.01)
            # 偏置初始化为小的随机值（正态分布）
            init.normal_(layer.bias, mean=0, std=0.01)

    def forward(self, x):
        """
        前向传播。

        参数:
            x (list or torch.Tensor): 输入数据，可以是 Python 列表或张量。

        返回:
            list: 模型的输出，转换为 Python 列表。
        """
        # 如果输入是列表，将其转换为张量
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # 转换为形状为 (1, input_dim) 的张量

        # 网络前向传播
        x = torch.relu(self.fc1(x))  # 隐藏层激活函数使用 ReLU
        x = self.fc2(x)  # 输出层无激活函数
        x = torch.softmax(x, dim=-1)

        # 将输出张量转换为 Python 列表
        return x.squeeze(0).tolist()  # 去掉 batch 维度并转换为列表

# Step 3: 加权计算主函数
def calculate_top_metrics(client, code_quality_analysis, llm_weight):

    total_weights = {}

    llm_result = get_llm_analysis(client, code_quality_analysis)
    # print(llm_result)

    # 第一部分权值：根据字符串在列表中的排序计算
    for i, string in enumerate(llm_result):
        # 权值从 1.0 到 0.1，线性递减
        order_weight = 1.0 - i * 0.08
        # 乘以 0.6
        total_weights[string] = order_weight * llm_weight
    # print(total_weights)
    # 准备神经网络输入
    all_metrics = [
        "语法检测", "可渲染性检测", "配置项完整检测",
        "图像与渲染截图的SSIM", "图像OCR检测的文字与配置项的余弦相似度",
        "图表类型均衡性", "配置项多样性",
        "代码重复", "图像重复", "联合重复",
        "数据量"
    ]

    X = [code_quality_analysis[metric] for metric in all_metrics]

    # 获取神经网络预测值
    predictor = MLP(input_dim=11, hidden_dim=20, output_dim=11)
    nn_scores = predictor(X)

    # 创建指标-得分映射
    metric_score_map = {metric: score for metric, score in zip(all_metrics, nn_scores)}
    # print(metric_score_map)

    for string, dict_weight in metric_score_map.items():
        total_weights[string] += dict_weight * (1-llm_weight)
    # print(total_weights)

    sorted_strings = sorted(total_weights.keys(), key=lambda x: total_weights[x], reverse=True)

    return  sorted_strings[:3]


if __name__ == "__main__":
    client = OpenAI(api_key="sk-3955d8823efd4f2483897446b91a7ffb", base_url="https://api.deepseek.com")
    # 使用示例
    sample_scores = {
        "语法检测": 0.45,
        "可渲染性检测": 0.32,
        "配置项完整检测": 0.32,
        "图像与渲染截图的SSIM": 0.18,
        "图像OCR检测的文字与配置项的余弦相似度": 0.81,
        "图表类型均衡性": 0.7,
        "配置项多样性": 0.3,
        "代码重复": 0.48,
        "图像重复": 0.22,
        "联合重复": 0.60,
        "数据量": 0.8
    }

    most_weakness = calculate_top_metrics(client=client, code_quality_analysis=sample_scores, llm_weight=0.7)  # 输出前三维度
    print(most_weakness)
