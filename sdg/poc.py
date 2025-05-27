"""PoC code to demonstrate the usage of the data_operator package.
"""

import os
import json

from .data_operator.operator import OperatorMeta
from . import data_operator
from .storage.dataset import Dataset, DataType, Datadir
from .task.task import Task
from .event import global_message_queue, EventType, EventResponse


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
    code_dir = Datadir('echart-code-sample', DataType.CODE)
    describe_data(code_dir)
    image_dir = Datadir('echart-image-sample', DataType.IMAGE)
    describe_data(image_dir)
    data_set = Dataset([code_dir, image_dir], 'echart-sample.metadata','key_configurations.md')
    describe_metadata(data_set.meta_path)
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="Load multimodal dataset done!"))


    result = data_set.evaluate_image_code_quality()

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="数据洞察发现靶点"))
    result = extract_secondary_metrics(result)
    # print(json.dumps(result, indent=4, ensure_ascii=False))
    # client = OpenAI(api_key="your key", base_url="https://api.deepseek.com")
    # calculate_top_metrics(client, result, 1)
    global_message_queue.put(EventResponse(event=EventType.REASONING, data="远端大模型分析..."))
    global_message_queue.put(EventResponse(event=EventType.REASONING, data="本地经验模型分析..."))
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="数据洞察发现靶点完成, 靶点为[数据量， 配置项多样性， 图像与渲染截图的SSIM]"))


    # build task workflow
    # 1st step: Randomly modify echarts configuration code to generate new code file
    # 2nd step: Generate echarts image for code without associated images
    # 3rd step: Generate echart configuration code for image without associated code
    # 4th step: Add noise to echarts image and generate new data pair
    task = Task(
        [
            registry['EChartMutationOperator'](), 
            registry['EchartsToImageOperator'](),
            registry['ImgToEchartsOperator'](), 
            registry['ImageRobustnessEnhancer']()
        ],
        data_set
        )
    task.run()

    result = data_set.evaluate_image_code_quality()

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
