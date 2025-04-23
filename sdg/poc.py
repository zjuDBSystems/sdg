"""PoC code to demonstrate the usage of the data_operator package.
"""

import pprint
import os
from textwrap import indent
from openai import OpenAI
from pkg_resources import ensure_directory
import json

from .data_operator.operator import Meta, Operator, OperatorMeta
from . import data_operator
from .storage.dataset import Dataset, DataType, Datadir, copy_dataset
from .task.task_type import TaskType
from .task.task import Task
from .event import global_message_queue, EventType, EventResponse
from .data_insights_identify import calculate_top_metrics


registry = OperatorMeta.get_registry()


def get_operators(data_type, task_type) -> list[Operator]:
    operators: list[Operator] = []
    for clas_name, cls in registry.items():
        if cls.accept(data_type, task_type):
            operators.append(clas_name)
    return operators


def describe_operator(name):
    operator: Operator = registry[name]
    meta: Meta = operator.get_meta()
    global_message_queue.put(EventResponse(event=EventType.REASONING, data=f"Operator name: {meta.name}"))
    global_message_queue.put(EventResponse(event=EventType.REASONING, data=f"Operator description: {meta.name}"))

def poc():
    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="List preprocessing operator!"))
    preprocessing_operators = get_operators(DataType.PYTHON,
                                            TaskType.PREPROCESSING)
    for operator_name in preprocessing_operators:
        describe_operator(operator_name)
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="List preprocessing operator done!"))
    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="List augmentation operator!"))
    augmentation_operators = get_operators(DataType.PYTHON,
                                           TaskType.AUGMENTATION)
    for operator_name in augmentation_operators:
        describe_operator(operator_name)
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="List augmentation operator done!"))

    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="Execute operators!"))
    dir = Datadir('raw', DataType.PYTHON)
    raw_dataset: Dataset = Dataset([dir], 'raw.metadata')

    dataset = raw_dataset
    preprocessing_task = Task([registry['PythonFormattingOperator']()],
                              TaskType.PREPROCESSING, dataset)
    preprocessing_task.run()

    dataset = preprocessing_task.final_dataset
    augmentation_task = Task([
        # registry['PythonReorderOperator'](),
        registry['PythonDocstringInsertOperator']()
    ], TaskType.AUGMENTATION, dataset)
    augmentation_task.run()
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="Execute operators done!"))


def img_aug():

    print('Augmentation operator----------------------------------------------')
    augmentation_operators = get_operators(DataType.IMAGE,
                                           TaskType.AUGMENTATION)
    for operator_name in augmentation_operators:
        describe_operator(operator_name)
    print('----------------------------------------------')

    img_dir = Datadir('raw/images', DataType.IMAGE)
    code_dir = Datadir('raw/echarts', DataType.ECHARTS)
    raw_dataset: Dataset = Dataset([img_dir, code_dir], 'raw/metadata.csv')

    augmentation_task = Task([
        registry['ImgToEchartsOperator'](),
    ], TaskType.AUGMENTATION, raw_dataset)
    augmentation_task.run()

def mutation_test():

    print('Augmentation operator----------------------------------------------')
    augmentation_operators = get_operators(DataType.ECHARTS,
                                           TaskType.AUGMENTATION)
    for operator_name in augmentation_operators:
        describe_operator(operator_name)
    print('----------------------------------------------')

    code_dir = Datadir('raw/echarts', DataType.ECHARTS)
    raw_dataset: Dataset = Dataset([code_dir], 'raw/metadata.csv')

    augmentation_task = Task([
        registry['EChartMutationOperator'](),
    ], TaskType.AUGMENTATION, raw_dataset)
    augmentation_task.run()

def add_noise_test():
    print('Augmentation operator----------------------------------------------')
    augmentation_operators = get_operators(DataType.IMAGE,
                                           TaskType.AUGMENTATION)
    for operator_name in augmentation_operators:
        describe_operator(operator_name)
    print('----------------------------------------------')

    img_dir = Datadir('raw/images', DataType.IMAGE)
    raw_dataset: Dataset = Dataset([img_dir], 'raw/metadata.csv')

    augmentation_task = Task([
        registry['ImageRobustnessEnhancer'](),
    ], TaskType.AUGMENTATION, raw_dataset)
    augmentation_task.run()

def total_aug_test():

    img_dir = Datadir('raw/images', DataType.IMAGE)
    code_dir = Datadir('raw/echarts', DataType.ECHARTS)
    raw_dataset: Dataset = Dataset([img_dir, code_dir], 'raw/metadata.csv')

    img_to_echarts_task = Task([
        registry['ImgToEchartsOperator'](
            api_key = ""
        ),
    ], TaskType.AUGMENTATION, raw_dataset)
    img_to_echarts_task.run()
    dataset = img_to_echarts_task.final_dataset
    
    augmentation_task = Task([
        registry['EChartMutationOperator'](),
    ], TaskType.AUGMENTATION, dataset)
    augmentation_task.run()
    dataset = augmentation_task.final_dataset

    add_noise_task = Task([
        registry['ImageRobustnessEnhancer'](),
    ], TaskType.AUGMENTATION, dataset)
    add_noise_task.run()

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
            registry['ImgToEchartsOperator'](), 
            registry['ImageRobustnessEnhancer']()],
        data_set
        )
    task.run()

if __name__ == '__main__':
    run_echart_task()