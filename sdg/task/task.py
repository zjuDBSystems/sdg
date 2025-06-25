"""Task module.

Typical usage example:
    
    task = Task(operators, TaskType.PREPROCESSING, DataType.PYTHON, dataset)
    task.run()
    final_dataset = task.final_dataset
"""

from uuid import UUID, uuid4
import time
import json
import timesfm
import re
import pandas as pd

from ..storage.dataset import Dataset, copy_dataset
from ..data_operator.operator import Operator
from ..event import global_message_queue, EventType, EventResponse
from collections import defaultdict
from typing import List, Callable, Iterable


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

class Task_SeriesForecast:
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
        self.model = self.load_series_forecast_model()

    def run(self) -> dict:
        """Executes the task by applying the operators to the input dataset."""
        dataset: Dataset = self.in_dataset
        result=None
        for operator in self.operators:
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f'执行制备算子 {operator.get_meta().name}'))
            print(f'执行制备算子 {operator.get_meta().name}')
            start = time.time()
            # dataset = copy_dataset(dataset) #
            operator.execute(dataset)
            self.out_datasets[operator.__class__.__name__] = dataset
            end = time.time()
            cost = end - start
            global_message_queue.put(EventResponse(event=EventType.REASONING, data=f'算子 {operator.get_meta().name} 执行完成! 耗时: {cost:.2f}秒'))
            print(f'算子 {operator.get_meta().name} 执行完成! 耗时: {cost:.2f}秒')
            # 时序数据预测，取消注释可执行
            # global_message_queue.put(EventResponse(event=EventType.REASONING, data="时序数据预测"))
            # print("时序数据预测")
            # start = time.time()
            # result = self.time_series_forecast(dataset)
            # end = time.time()
            # cost = end - start
            # global_message_queue.put(EventResponse(event=EventType.REASONING, data=f"时序数据预测完成, 耗时: {cost:.2f}秒"))
            # print(f"时序数据预测完成, 耗时: {cost:.2f}秒")
            # global_message_queue.put(EventResponse(event=EventType.REASONING, data=json.dumps(result, indent=4, ensure_ascii=False)))
        self.final_dataset = dataset
        return result

    def load_series_forecast_model(self):
        global_message_queue.put(EventResponse(event=EventType.REASONING, data="时序模型加载"))
        print("时序模型加载")
        start = time.time()
        model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                horizon_len=128,
                num_layers=50,
                context_len=2048,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
        )
        end = time.time()
        cost = end - start
        global_message_queue.put(EventResponse(event=EventType.REASONING, data=f"时序模型加载完成, 耗时: {cost:.2f}秒"))
        print(f"时序模型加载完成, 耗时: {cost:.2f}秒")

        return model

    def time_series_forecast(self, dataset: Dataset) -> List:

        df = pd.read_csv(dataset.dirs[0].data_path, na_values=['nan', 'None', ''])

        batch_size = 32
        context_len = 96
        horizon_len = 96
        total_len = context_len + horizon_len

        target_patterns = '延安发电1号机组'

        cov_num_patterns = [
            "负荷预测","风电总出力预测数值","光伏总出力预测数值","新能源总出力预测数值","非市场机组总出力预测","外来电交易计划","竞价空间",
            "elev","az","freq_负荷预测","freq_风电总出力预测数值","freq_光伏总出力预测数值","freq_新能源总出力预测数值",
            "freq_非市场机组总出力预测","freq_外来电交易计划","freq_竞价空间"
        ]
        cov_cat_patterns = [
            "month","day","weekday","hour","minute","holiday","延安发电1号机组运行状态"
        ]

        input_data, get_dynamic_numerical_covariates, get_dynamic_categorical_covariates = (self.get_batched_data_fn(df,
            batch_size=batch_size,context_len=context_len,horizon_len=horizon_len, target_col=target_patterns,
            cov_num_patterns=cov_num_patterns,cov_cat_patterns=cov_cat_patterns))
        metrics = defaultdict(list)

        # Define metrics
        def mse(y_pred, y_true):
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            return np.mean(np.square(y_pred - y_true), axis=1, keepdims=True)

        def mae(y_pred, y_true):
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)
            return np.mean(np.abs(y_pred - y_true), axis=1, keepdims=True)

        def mape(y_pred, y_true):
            pred = np.array(y_pred)
            true = np.array(y_true)

            abs_error = np.abs(pred - true)

            denominator = np.abs(true)
            safe_denominator = np.where(denominator == 0, 600, denominator)

            percentage_error = abs_error / safe_denominator
            return np.mean(percentage_error, axis=1, keepdims=True)

        import numpy as np

        def accuracy_within_radius(y_pred, y_true, radius=50):
            y_pred = np.array(y_pred)
            y_true = np.array(y_true)

            abs_diff = np.abs(y_pred - y_true)

            within_radius = abs_diff <= radius

            return np.mean(within_radius, axis=1, keepdims=True)

        for i, example in enumerate(input_data()):
            raw_forecast, _ = self.model.forecast(
                inputs=example["inputs"], freq=[0] * len(example["inputs"])
            )
            start_time = time.time()

            cov_forecast, ols_forecast = self.model.forecast_with_covariates(
                inputs=example["inputs"],
                dynamic_numerical_covariates=get_dynamic_numerical_covariates(example),
                dynamic_categorical_covariates=get_dynamic_categorical_covariates(example),
                static_numerical_covariates={},
                static_categorical_covariates={},
                freq=[0] * len(example["inputs"]),
                xreg_mode="xreg + timesfm",
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=False,
            )
            print(
                f"\rFinished batch {i} linear in {time.time() - start_time} seconds",
                end="",
            )
            metrics["eval_mae_timesfm"].extend(
                mae(raw_forecast[:, :horizon_len], example["outputs"])
            )
            metrics["eval_mae_xreg_timesfm"].extend(mae(cov_forecast, example["outputs"]))
            metrics["eval_mae_xreg"].extend(mae(ols_forecast, example["outputs"]))

            metrics["eval_mse_timesfm"].extend(
                mse(raw_forecast[:, :horizon_len], example["outputs"])
            )
            metrics["eval_mse_xreg_timesfm"].extend(mse(cov_forecast, example["outputs"]))
            metrics["eval_mse_xreg"].extend(mse(ols_forecast, example["outputs"]))

            metrics["eval_mape_timesfm"].extend(
                mape(raw_forecast[:, :horizon_len], example["outputs"])
            )
            metrics["eval_mape_xreg_timesfm"].extend(mape(cov_forecast, example["outputs"]))
            metrics["eval_mape_xreg"].extend(mape(ols_forecast, example["outputs"]))

            metrics["eval_accuracy_timesfm"].extend(
                accuracy_within_radius(raw_forecast[:, :horizon_len], example["outputs"])
            )
            metrics["eval_accuracy_xreg_timesfm"].extend(accuracy_within_radius(cov_forecast, example["outputs"]))
            metrics["eval_accuracy_xreg"].extend(accuracy_within_radius(ols_forecast, example["outputs"]))

        for k, v in metrics.items():
            print(f"{k}: {np.mean(v)}")

    def get_batched_data_fn(
            self,
            df: pd.DataFrame,
            *,
            target_col: str,
            cov_num_patterns: Iterable[str] | str | None = None,
            cov_cat_patterns: Iterable[str] | str | None = None,
            batch_size: int = 32,
            context_len: int = 96,
            horizon_len: int = 96,
    ) -> tuple[Callable[[], dict], Callable[[dict], dict], Callable[[dict], dict]]:

        def _expand_patterns(patterns) -> List[str]:
            if patterns is None:
                return []
            # 允许直接给 str
            if isinstance(patterns, str):
                patterns = [patterns]

            matched: list[str] = []
            for pat in patterns:
                regex = re.compile(pat)
                matched.extend([c for c in df.columns if regex.fullmatch(c) or regex.search(c)])
            # 去重同时保持第一次出现的顺序
            seen = set()
            uniq = [c for c in matched if not (c in seen or seen.add(c))]
            return uniq

        covariate_numerical_cols = _expand_patterns(cov_num_patterns)
        covariate_categorical_cols = _expand_patterns(cov_cat_patterns)

        examples = defaultdict(list)
        total_len = context_len + horizon_len

        for start in range(0, len(df) - total_len, horizon_len):
            ctx_end = start + context_len

            examples["inputs"].append(df[target_col].iloc[start:ctx_end].tolist())
            examples["outputs"].append(df[target_col].iloc[ctx_end: ctx_end + horizon_len].tolist())

            for col in covariate_numerical_cols:
                examples[col].append(df[col].iloc[start: ctx_end + horizon_len].tolist())

            for col in covariate_categorical_cols:
                examples[col].append(df[col].iloc[start: ctx_end + horizon_len].tolist())

        num_examples = len(examples["inputs"])

        get_dynamic_numerical_covariates = (
            lambda ex: {c: ex[c] for c in covariate_numerical_cols}
        )
        get_dynamic_categorical_covariates = (
            lambda ex: {c: ex[c] for c in covariate_categorical_cols}
        )

        def data_fn():
            for i in range(0, num_examples, batch_size):
                yield {k: v[i: i + batch_size] for k, v in examples.items()}

        return data_fn, get_dynamic_numerical_covariates, get_dynamic_categorical_covariates