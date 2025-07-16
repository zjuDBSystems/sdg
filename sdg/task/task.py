"""Task module.

Typical usage example:
    
    task = Task(operators, TaskType.PREPROCESSING, DataType.PYTHON, dataset)
    task.run()
    final_dataset = task.final_dataset
"""

import os
from uuid import UUID, uuid4
import time
import json
import re
import pandas as pd

from ..storage.dataset import Dataset, copy_dataset
from ..data_operator.operator import Operator
from ..event import global_message_queue, EventType, EventResponse
from collections import defaultdict
from typing import List, Callable, Iterable
import pickle as pkl


class Task:
    """Represents a task that processes a dataset through a series of operators.

    Attributes:
        operators: A list of operators to be applied to the dataset.
        task_type: The type of the task.
        data_type: The type of data being processed.
        in_dataset: The initial dataset to be processed.
        id: A unique identifier for the task.
        out_datasets: A dictionary mapping operator names to their
        output datasets.
        final_dataset: The final dataset after all operators have been 
        applied.

    """

    def __init__(self, operators: list[Operator],
                 in_dataset: Dataset):
        """Initializes a task with the given operators and input dataset.

        Args:
            operators: A list of operators to be applied to the dataset.
            task_type: The type of the task.
            in_dataset: The initial dataset to be processed.
        """
        self.operators: list[Operator] = operators
        self.in_dataset: Dataset = in_dataset
        self.id: UUID = uuid4()
        self.out_datasets: dict[str, Dataset] = {}
        self.final_dataset: Dataset | None = None

    def run(self) -> dict:
        """Executes the task by applying the operators to the input dataset."""
        dataset: Dataset = self.in_dataset
        result=None
        for operator in self.operators:
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f'执行制备算子 {operator.get_meta().name}'))
            start = time.time()
            dataset = copy_dataset(dataset)
            operator.execute(dataset)
            self.out_datasets[operator.__class__.__name__] = dataset
            end = time.time()
            cost = end - start
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f'算子 {operator.get_meta().name} 执行完成! 耗时: {cost:.2f}秒'))
            global_message_queue.put(EventResponse(event=EventType.REASONING, data="数据质量评估"))
            start = time.time()
            result = dataset.evaluate_image_code_quality()
            end = time.time()
            cost = end - start
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f"数据质量评估完成, 耗时: {cost:.2f}秒"))
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=json.dumps(result, indent=4, ensure_ascii=False)))
        self.final_dataset = dataset
        return result

class Task_power:
    """Represents a task that processes a dataset through a series of operators.

    Attributes:
        operators: A list of operators to be applied to the dataset.
        task_type: The type of the task.
        data_type: The type of data being processed.
        in_dataset: The initial dataset to be processed.
        id: A unique identifier for the task.
        out_datasets: A dictionary mapping operator names to their
        output datasets.
        final_dataset: The final dataset after all operators have been 
        applied.

    """

    def __init__(self, operators: list[Operator],
                 in_dataset: Dataset):
        """Initializes a task with the given operators and input dataset.

        Args:
            operators: A list of operators to be applied to the dataset.
            task_type: The type of the task.
            in_dataset: The initial dataset to be processed.
        """
        self.operators: list[Operator] = operators
        self.in_dataset: Dataset = in_dataset
        self.id: UUID = uuid4()
        self.out_datasets: dict[str, Dataset] = {}
        self.final_dataset: Dataset | None = None

    def run(self) -> dict:
        """Executes the task by applying the operators to the input dataset."""
        dataset: Dataset = self.in_dataset

        input_datafile_ls = ['shanxi_day_train_96_96.pkl', 'shanxi_day_train_192_192.pkl', 'shanxi_day_train_384_384.pkl']
        output_datafile_ls = ['shanxi_day_train_total_96_96.pkl', 'shanxi_day_train_total_192_192.pkl', 'shanxi_day_train_total_384_384.pkl']
        for input_datafile, output_datafile in zip(input_datafile_ls, output_datafile_ls):
            input_datafile_path = os.path.join(dataset.dirs[0].data_path, input_datafile)
            ls_df = pkl.load(open(input_datafile_path, "rb"))
            with open(os.path.join(dataset.dirs[0].data_path, output_datafile), "wb") as file:
                pkl.dump(ls_df, file, protocol=5)

        result=None
        for operator in self.operators:
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f'执行制备算子 {operator.get_meta().name}'))
            start = time.time()
            # dataset = copy_dataset(dataset)
            operator.execute(dataset)
            self.out_datasets[operator.__class__.__name__] = dataset
            end = time.time()
            cost = end - start
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f'算子 {operator.get_meta().name} 执行完成! 耗时: {cost:.2f}秒'))
            global_message_queue.put(EventResponse(event=EventType.REASONING, data="数据质量评估"))
            start = time.time()
            result = dataset.evaluate_table_quality()
            end = time.time()
            cost = end - start
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f"数据质量评估完成, 耗时: {cost:.2f}秒"))
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=json.dumps(result, indent=4, ensure_ascii=False)))
        self.final_dataset = dataset
        return result

