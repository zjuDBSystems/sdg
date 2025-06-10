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
你将收到一份代码数据集的质量评估报告。该报告包含 **9个质量评估维度**，每个维度都有一个分数（Score0，范围为 1-100），用于衡量数据集在该维度上的质量表现。对于每一个维度，Score越高，表示该维度的质量越好。

你的任务是根据这些维度的分数，分析它们对整体数据质量的影响程度，这个影响程度也是一个分数（Score1，范围为 1-100）。Score1和Score0相反，Score1越高，表示这个维度表现越差，越是当前数据的缺点。（所以Score1和Score0一定程度成反比）。Score1需要综合考虑Score0和维度重要性。

维度越重要，Score1越高；Score0越高，Score1越低。

比如，数据量的Score1为40，这是一个比较低的分数，而数据量又是一个很重要的指标，所以我们的Score1会给出90.

你需要返回一个 Python 字典。字典中应包括这 9 个维度（9个维度为字典的key），每个key对应的value是这个维度的Score1。

请注意，Score0需要综合考虑Score1和维度重要性。Score0越高表示这个维度越有可能成为重大缺点。
---

### **输入说明**
输入是一个包含 9 个维度及其对应分数的字典，格式如下：

```python
{{
"语法检测": 85,
"缺失率": 92,
"配置项完整检测": 78,
"图像与渲染截图的匹配度": 88,
"图表类型均衡性": 75,
"配置项多样性": 93,
"代码重复": 68,
"图像重复": 72,
"数据量": 89
}}
```

---

### **任务要求**
1. **分析维度重要性**：
- 根据每个维度的分数，结合其对整体数据质量的影响程度，判断哪些维度更重要。

2. **输出格式**：
- 返回一个 Python 字典，字典中包含 9 个维度名称，和他们的整体得分。
- 输出必须严格遵循以下格式：

```python
{{
"语法检测": 87,
"缺失率": 95,
"配置项完整检测": 74,
"图像与渲染截图的匹配度": 79,
"图表类型均衡性": 70,
"配置项多样性": 85,
"代码重复": 65,
"图像重复": 68,
"数据量": 84
}}
```

---

### **示例**
假设输入如下：

```python
{{
"语法检测": 85,
"缺失率": 92,
"配置项完整检测": 78,
"图像与渲染截图的匹配度": 88,
"图表类型均衡性": 75,
"配置项多样性": 93,
"代码重复": 68,
"图像重复": 72,
"数据量": 89
}}
```

经过分析后，输出应为：

```python
{{
"语法检测": 30,
"缺失率": 55,
"配置项完整检测": 60,
"图像与渲染截图的匹配度": 40,
"图表类型均衡性": 30,
"配置项多样性": 29,
"代码重复": 80,
"图像重复": 79,
"数据量": 40
}}
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

            print("LLM 的整体维度得分：")
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

def generate_detailed_analysis(client, code_quality_analysis, sorted_metrics, max_retries=5):
    retries = 0
    while retries < max_retries:
        try:
            # 构造提示词
            prompt = f"""### **任务描述**
你将收到一份代码数据集的质量评估报告。该报告包含 **9个质量评估维度**，每个维度都有一个分数（范围为 1-100），用于衡量数据集在该维度上的质量表现。所有维度的分数越高，表示该维度的质量越好。

你还会收到一份基于这个质量评估报告做的维度排序。维度排序越靠前，越说明这个指标在这个数据集中，因为**维度定义**和**维度分数**对数据质量有重要的影响。

你的任务是对这些指标返回指标的定义，以及如果这个指标分数低，会对LLM训练带来什么影响。


---

### **输入说明**
输入是一个包含 10 个维度及其对应分数的字典，格式如下：

```python
{{
"语法检测": 85,
"缺失率": 92,
"配置项完整检测": 78,
"图像与渲染截图的匹配度": 88,
"图表类型均衡性": 75,
"配置项多样性": 93,
"代码重复": 68,
"图像重复": 72,
"数据量": 89
}}
```

一个维度排序，格式如下：
["配置项多样性", "缺失率", "数据量", "图像与渲染截图的匹配度", "语法检测", "配置项完整检测", "图像重复", "图表类型均衡性", "代码重复"]
---

### **任务要求**
1. **返回指标的定义，以及如果这个指标分数低，会对LLM训练带来什么影响**：

2. **输出格式**：
- 返回一个 Python 字典，列表中包含 9 个维度名称，说明每个维度的定义和影响。
- 输出必须严格遵循以下格式：

```python
{{
    ”维度1“: "定义以及影响"
}}
```

### **你的任务**
{json.dumps(code_quality_analysis, indent=4, ensure_ascii=False)}

{json.dumps(sorted_metrics, indent=4, ensure_ascii=False)}
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

            print("LLM 的靶点解释：")
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
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        """
        初始化一个简单的多层感知机（MLP）模型。

        参数:
            input_dim (int): 输入维度，默认为 10。
            hidden_dim (int): 隐藏层维度，默认为 20。
            output_dim (int): 输出维度，默认为 10。
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
def sort_metrics(client, code_quality_analysis, llm_weight):

    llm_result = get_llm_analysis(client, code_quality_analysis)
    # print(llm_result)

    # 第一部分权值：根据字符串在列表中的排序计算
    sorted_llm_result = sorted(llm_result.items(), key=lambda item: item[1], reverse=True)


    sorted_llm_result = dict(sorted_llm_result)

    total_weights = sorted_llm_result

    # 准备神经网络输入
    all_metrics = [
        "语法检测", "配置项完整检测",
        "图像与渲染截图的匹配度",
        "图表类型均衡性", "配置项多样性",
        "代码重复", "图像重复", "缺失率",
        "数据量"
    ]

    X = [code_quality_analysis[metric] for metric in all_metrics]

    # 获取神经网络预测值
    predictor = MLP(input_dim=9, hidden_dim=20, output_dim=9)
    nn_scores = predictor(X)

    # 创建指标-得分映射
    metric_score_map = {metric: score for metric, score in zip(all_metrics, nn_scores)}
    # print(metric_score_map)

    for string, dict_weight in metric_score_map.items():
        total_weights[string] += (dict_weight * 100) * (1-llm_weight)
    # print(total_weights)

    sorted_strings = sorted(total_weights.keys(), key=lambda x: total_weights[x], reverse=True)

    sorted_total_weights = dict(sorted(total_weights.items(), key=lambda item: item[1], reverse=True))

    return  sorted_strings, sorted_total_weights


if __name__ == "__main__":
    client = OpenAI(api_key="sk-3955d8823efd4f2483897446b91a7ffb", base_url="https://api.deepseek.com")
    # 使用示例
    sample_scores = {
        '语法检测': 84.53, '配置项完整检测': 93.74, '图像与渲染截图的匹配度': 82.6, '缺失率': 75.0, '图表类型均衡性': 66.65, '配置项多样性': 44,
        '代码重复': 70.0, '图像重复': 70.0, '数据量': 40.44
    }

    sorted_metrics, sorted_total_weights = sort_metrics(client=client, code_quality_analysis=sample_scores, llm_weight=0.7)
    print("最终靶点排序")
    print(sorted_metrics)
    analysis = generate_detailed_analysis(client, code_quality_analysis=sample_scores, sorted_metrics=sorted_metrics)
    print("最终分析")
    print(analysis)
