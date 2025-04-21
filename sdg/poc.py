"""PoC code to demonstrate the usage of the data_operator package.
"""

import pprint

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



if __name__ == '__main__':
    # poc()
    # img_aug()
    # mutation_test()
    # add_noise_test()
    total_aug_test()