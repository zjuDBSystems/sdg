"""PoC code to demonstrate the usage of the data_operator package.
"""

import os
import json
import time

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
    table_dir = Datadir('shanxi-power-table/shanxi_day.csv', DataType.TABLE)
    data_set = Dataset([table_dir], '', 'key_configurations.md')
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="Load power table dataset done!"))

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据质量评估"))
    start = time.time()
    result = data_set.evaluate_table_quality()
    global_message_queue.put(
        EventResponse(event=EventType.RESPONSE, data="数据质量评估完成, 耗时: {:.2f}秒".format(time.time() - start)))
    global_message_queue.put(
        EventResponse(event=EventType.REASONING, data=json.dumps(result, indent=4, ensure_ascii=False)))

    def target_discovery(client, result):
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "数据:" + json.dumps(result, indent=4, ensure_ascii=False) + "请根据上述数据，找出较为重要(>=50)的一级指标。\n请参照以下格式：\n<AI>数据洞察发现靶点完成, 靶点为[数据量, 多尺度增强...]\n<AI>"}
            ],
            stream=False,
        )
        return response.choices[0].message.content

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据洞察发现靶点"))
    # client = OpenAI(api_key="your key", base_url="https://api.deepseek.com")
    # result = target_discovery(client, result)
    # global_message_queue.put(EventResponse(event=EventType.REASONING, data="远端大模型分析..."))
    # global_message_queue.put(EventResponse(event=EventType.RESPONSE, data=result))
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="数据洞察发现靶点完成, 靶点为[主频提取, 时序特征增强, 多尺度增强, 数据量]"))

    task = Task_SeriesForecast(
        [
            # 提取主频成分作为协变量进行数据增强
            registry['FrequencyEnhanceOperator'](),
            # 数据层面进行非平稳处理
            registry['NonStationaryProcessOperator'](),
            # 提取时间特征作为协变量进行数据增强
            registry['TimeFeatureEnhanceOperator'](),
            # 对特定变量添加多尺度特征进行数据增强
            registry['MultiscaleEnhanceOperator'](),
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

    # result = data_set.evaluate_table_quality()


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

if __name__ == '__main__':
    # run_echart_task()
    # code_to_img()
    # img_aug()
    # aug_process()
    run_power_task()

