"""PoC code to demonstrate the usage of the data_operator package.
"""

import pprint
import os

from .data_operator.operator import Meta, Operator, OperatorMeta
from . import data_operator
from .storage.dataset import Dataset, DataType, Datadir
from .task.task_type import TaskType
from .task.task import Task
from .event import global_message_queue, EventType, EventResponse


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

def run_echart_task():
    global_message_queue.put(EventResponse(event=EventType.REQUEST, data="Load multimodal dataset, include code and image!"))
    code_dir = Datadir('echart-code', DataType.CODE)
    describe_data(code_dir)
    image_dir = Datadir('echart-image', DataType.IMAGE)
    describe_data(image_dir)
    data_set = Dataset([code_dir, image_dir], 'echart.metadata')
    describe_metadata(data_set.meta_path)
    global_message_queue.put(EventResponse(event=EventType.RESPONSE, data="Load multimodal dataset done!"))

if __name__ == '__main__':
    run_echart_task()