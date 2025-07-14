"""PoC code to demonstrate the usage of the data_operator package.
"""

import os
import json
import time

import pandas as pd
from openai import OpenAI

from sdg.data_operator.operator import OperatorMeta
from sdg import data_operator
from sdg.storage.dataset import Dataset, DataType, Datadir
from sdg.task.task import Task, Task_SeriesForecast
from sdg.event import global_message_queue, EventType, EventResponse


registry = OperatorMeta.get_registry()

def describe_data(datadir: Datadir):
    dir_path = datadir.data_path
    count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    data_type = datadir.data_type.value
    global_message_queue.put(EventResponse(EventType.REASONING, f'{data_type} data in {dir_path} has {count} files!'))

def describe_metadata(metadata_path: str):
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    global_message_queue.put(EventResponse(EventType.REASONING, f'multimodal dataset contains {len(lines) - 1} data pairs!'))

def extract_secondary_metrics(result):
    secondary_metrics = {}
    for category, metrics in result["二级指标"].items():
        secondary_metrics.update(metrics)
    return secondary_metrics

def run_echart_task():
    # load echart example dataset
    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="Load multimodal dataset, include code and image!"))
    code_dir = Datadir('dirty-echart-code', DataType.CODE)
    describe_data(code_dir)
    image_dir = Datadir('dirty-echart-image', DataType.IMAGE)
    describe_data(image_dir)
    data_set = Dataset([code_dir, image_dir], 'dirty-echart.metadata','key_configurations.md')
    describe_metadata(data_set.meta_path)
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="Load multimodal dataset done!"))


    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据质量评估"))
    start = time.time()
    result = data_set.evaluate_image_code_quality()
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="数据质量评估完成, 耗时: {:.2f}秒".format(time.time() - start)))
    global_message_queue.put(EventResponse(event=EventType.REASONING, data=json.dumps(result, indent=4, ensure_ascii=False)))

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据洞察发现靶点"))
    # result = extract_secondary_metrics(result)
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    # client = OpenAI(api_key="your key", base_url="https://api.deepseek.com")
    # calculate_top_metrics(client, result, 1)
    global_message_queue.put(EventResponse(event=EventType.REASONING, data="远端大模型分析..."))
    global_message_queue.put(EventResponse(event=EventType.REASONING, data="本地经验模型分析..."))
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="数据洞察发现靶点完成, 靶点为[数据量， 配置项多样性， 图像与渲染截图的SSIM]"))


    # build task workflow

    task = Task(
        [
            # 配置项修正
            registry['ConfigAmendOperator'](),
            # 语法修正
            registry['SyntaxAmendOperator'](),
            # 配置项多样性
            registry['DiversityEnhanceOperator'](
                api_key = "api_key"
            ),
            # 图像的echarts代码补全
            registry['ImgToEchartsOperator'](
                api_key = "api_key"
            ),
            # echarts代码随机变异(生成新的突变代码，此步骤只生成代码，没有生成相应的图像)
            registry['EChartMutationOperator'](),
            # echarts代码的图像补全
            registry['EchartsToImageOperator'](),
            # 图像随机扰动
            # registry['ImageRobustnessEnhancer'](),
            ],
            data_set
        )
    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="开始执行任务流程"))
    start = time.time()
    task.run()
    end = time.time()
    cost = end - start
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="任务流程执行完成, 耗时: {:.2f}秒".format(cost)))

    result = data_set.evaluate_image_code_quality()

def run_power_task():
    # load power example dataset
    global_message_queue.put(
        EventResponse(event=EventType.REQUEST, data="Load power table dataset!"))
    table_dir = Datadir('shanxi-power-table', DataType.TABLE)
    data_set = Dataset([table_dir], '', 'key_configurations.md')
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="Load power table dataset done!"))

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据质量评估"))
    start = time.time()
    result = data_set.evaluate_table_quality("shanxi_day_train_96_96.pkl")
    global_message_queue.put(
        EventResponse(event=EventType.RESPONSE, data="数据质量评估完成, 耗时: {:.2f}秒".format(time.time() - start)))
    global_message_queue.put(
        EventResponse(event=EventType.REASONING, data=json.dumps(result, indent=4, ensure_ascii=False)))

    secondary_weights = {
        "数据量": {
            "时间粒度覆盖率": 0.3,
            "趋势强度": 0.05,
            "季节性强度": 0.3,
            "主频强度": 0.25,
            "样本均衡性": 0.1
        },
        "数据内在质量": {
            "数据完整性": 0.9,
            "标签一致性": 0.1
        },
        "数据表示质量": {
            "时序平稳性": 0.6,
            "时间特征完备度": 0.4
        },
        "数据上下文质量": {
            "领域知识多样性": 0.3,
            "领域知识完整性": 0.7
        },
        "数据冗余": {
            "样本均衡性": 0.4,
            "特征独立性": 0.6
        }
    }

    def target_discovery(client, result):
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "数据:" + result + "任务：基于电价表格时间序列数据，进行未来电价时间序列预测；上述输入数据的每一项中的每一列分别为：一级指标,，二级指标(靶点)，重要性，当前分数，问题严重性，迫切性。请对上述二级指标(靶点)按照迫切性指标进行按照由大到小的顺序进行排序。\n请参照以下格式(请严格参照，不要添加多余的内容)：\n数据洞察发现靶点完成, 靶点为[<二级指标1>: <指标数值1>, <二级指标2>: <指标数值2>, <二级指标3>: <指标数值3>]\n"}
            ],
            stream=False,
        )
        return response.choices[0].message.content

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据洞察发现靶点"))

    indicator_relations = define_indicator_relations(result)
    urgency_df = calculate_urgency(result, indicator_relations, secondary_weights)

    client = OpenAI(api_key="sk-2694f692c8a74876a7a8856fdaf7ed7e", base_url="https://api.deepseek.com")
    result = target_discovery(client, str(urgency_df.values.tolist()))
    global_message_queue.put(EventResponse(event=EventType.REASONING, data="远端大模型分析..."))
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data=result))
    print(result)

    task = Task_SeriesForecast(
        [
            # 提取主频成分作为协变量进行数据增强
            # registry['FrequencyEnhanceOperator'](),
            # 数据层面进行非平稳处理
            # registry['NonStationaryProcessOperator'](),
            # 提取时间特征作为协变量进行数据增强
            # registry['TimeFeatureEnhanceOperator'](),
            # 对特定变量添加多尺度特征进行数据增强
            # registry['MultiscaleEnhanceOperator'](),
        ],
        data_set
    )

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="开始执行任务流程"))
    print("开始执行任务流程")
    start = time.time()
    task.run()
    end = time.time()
    cost = end - start
    global_message_queue.put(
        EventResponse(event=EventType.RESPONSE, data="任务流程执行完成, 耗时: {:.2f}秒".format(cost)))
    print("任务流程执行完成, 耗时: {:.2f}秒".format(cost))

    result = data_set.evaluate_table_quality("shanxi_day_train_total.pkl")
    # print(json.dumps(result, indent=4, ensure_ascii=False))


def data_evaluation():
    negative_code_dir = Datadir('echart-code-sample-negative', DataType.CODE)
    negative_image_dir = Datadir('echart-image-sample-negative', DataType.IMAGE)
    negative_dataset = Dataset([negative_code_dir, negative_image_dir], 'echart-sample-negative.metadata','key_configurations.md')
    result = negative_dataset.evaluate_image_code_quality()
    print(json.dumps(result, indent=4, ensure_ascii=False))

    positive_code_dir = Datadir('echart-code-sample-positive', DataType.CODE)
    positive_image_dir = Datadir('echart-image-sample-positive', DataType.IMAGE)
    positive_dataset = Dataset([positive_code_dir, positive_image_dir], 'echart-sample-positive.metadata','key_configurations.md')
    result = positive_dataset.evaluate_image_code_quality()
    print(json.dumps(result, indent=4, ensure_ascii=False))

def aug_process():
    start_time = time.time()  # 记录开始时间
    negative_code_dir = Datadir('dirty-echart-code', DataType.CODE)
    negative_image_dir = Datadir('dirty-echart-image', DataType.IMAGE)
    negative_dataset = Dataset([negative_code_dir, negative_image_dir], 'dirty-echart.metadata','key_configurations.md')
    augmentation_task = Task([
        # 配置项修正
        registry['ConfigAmendOperator'](
            score_file = './per_scores.csv',
        ),
        # 语法修正
        registry['SyntaxAmendOperator'](
            score_file = './per_scores.csv',
        ),
        # registry['SyntaxAmendOperatorGPT'](
        #     api_key = "api_key",
        #     score_file = './per_scores.csv',
        # ),
        # 配置项多样性 1
        # registry['DiversityAmendOperator'](
        #     api_key = "api_key"
        # ),
        # 配置项多样性 2
        registry['DiversityEnhanceOperator'](
            api_key = "api_key",
            score_file = './per_scores.csv',
        ),
        # 图像的echarts代码补全
        registry['ImgToEchartsOperator'](
            api_key = "api_key"
        ),
        # echarts代码随机变异(生成新的突变代码，此步骤只生成代码，没有生成相应的图像)
        registry['EChartMutationOperator'](),
        # echarts代码的图像补全
        registry['EchartsToImageOperator'](),
        # 图像随机扰动
        registry['ImageRobustnessEnhancer'](),


    ], negative_dataset)
    augmentation_task.run()
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算执行时间
    new_dataset = augmentation_task.final_dataset
    print(f"数据集路径{new_dataset.dirs}")
    print(f"数据集metadata路径{new_dataset.meta_path}")
    result = new_dataset.evaluate_image_code_quality()
    print(json.dumps(result, indent=4, ensure_ascii=False))
    print(f"代码运行时间: {execution_time} 秒")

def define_indicator_relations(scores_data):
    return {primary: list(secondaries.keys()) for primary, secondaries in scores_data["二级指标"].items()}

def calculate_urgency(scores_data, indicator_relations, secondary_weights=None):
    ideal_value = 100.0
    results = []

    # 提取所有二级指标的当前分数
    current_secondary_scores = {}
    for primary, secondaries in scores_data["二级指标"].items():
        for secondary_name, score in secondaries.items():
            # 处理重复的指标，如“样本均衡性”
            if secondary_name not in current_secondary_scores:
                current_secondary_scores[secondary_name] = score

    for primary, secondaries in indicator_relations.items():
        # 1. 获取权重分配
        if secondary_weights and primary in secondary_weights:
            weights = secondary_weights[primary]
            # 若权重未覆盖所有二级指标，自动补齐均分
            total_weight = sum(weights.get(sec, 0) for sec in secondaries)
            if total_weight == 0:
                # 全为0，均分
                weights = {sec: 1/len(secondaries) for sec in secondaries}
            else:
                # 归一化
                weights = {sec: weights.get(sec, 0)/total_weight for sec in secondaries}
        else:
            # 默认均分
            weights = {sec: 1/len(secondaries) for sec in secondaries}

        for secondary_name in secondaries:
            # 2. 计算问题严重性分数
            current_val = current_secondary_scores.get(secondary_name, 0)
            # 公式: (理想值 - 当前值) / 理想值
            problem_level = (ideal_value - current_val) / ideal_value

            # 3. 计算最终迫切性分数
            importance_score = weights[secondary_name]
            urgency_score = importance_score * problem_level

            results.append({
                "一级指标": primary,
                "二级指标(靶点)": secondary_name,
                "重要性": importance_score,
                "当前分数": current_val,
                "问题严重性": problem_level,
                "迫切性": urgency_score,
            })

    return pd.DataFrame(results)


if __name__ == '__main__':
    # run_echart_task()
    # code_to_img()
    # img_aug()
    # aug_process()
    run_power_task()

