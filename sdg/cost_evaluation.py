
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable
import random
from .storage.dataset import Dataset, DataType, Datadir, copy_dataset





operator_to_metrics = {
    "ConfigAmendOperator": ["配置项完整检测"],
    "DiversityEnhanceOperator": ["配置项多样性"],
    "EChartMutationOperator": ["数据量", "配置项多样性"],
    "EchartsToImageOperator": ["缺失率"],
    "ImageRobustnessEnhancer": ["数据量", "图像重复"],
    "ImgToEchartsOperator": ["缺失率"],
    "SyntaxAmendOperatorGPT": ["语法检测"]
}



def flatten_secondary_metrics(result: dict) -> dict:
    """
    将 result['二级指标'] 中的嵌套结构扁平化为一个单层 dict，形如 sample_scores。
    """
    sample_scores = {}

    secondary = result.get("二级指标", {})
    for primary_category, sub_metrics in secondary.items():
        for metric_name, score in sub_metrics.items():
            # 统一字段命名（选填：去除重复前缀等）
            sample_scores[metric_name] = score
    if "缺失率得分" in sample_scores:
        sample_scores["缺失率"] = sample_scores.pop("缺失率得分")
    return sample_scores



@dataclass
class OperatorData:
    name: str
    ti: float  # time per record
    ri: int    # number of records to process
    ci: float  # raw cost input (e.g., tokens or CPU hours)
    wi: float  # target weight
    type: str  # 'LLM' or 'CPU'


def refresh_operator_costs(
    registry: Dict[str, type],
    operator_pool: List[OperatorData],
    dataset: Dataset,
    sorted_total_weights: Dict[str, float],
    operator_to_metrics: Dict[str, List[str]]
) -> List[OperatorData]:
    updated_pool = []

    # 从原有 pool 提取算子名称集合
    pool_names = {op.name for op in operator_pool}

    for cls in registry.values():
        try:
            instance = cls()
            cost_info = instance.get_cost(dataset)
            name = cost_info["name"]

            # 跳过非目标算子
            if name not in pool_names:
                continue

            # 获取该算子对应的指标列表
            metrics = operator_to_metrics.get(name, [])

            # 计算该算子的综合权重（wi）
            weight = sum(sorted_total_weights.get(metric, 0.0) for metric in metrics)

            op_data = OperatorData(
                name=name,
                ti=cost_info["ti"],
                ri=cost_info["ri"],
                ci=cost_info["ci"],
                wi=weight,
                type=cost_info["type"]
            )
            updated_pool.append(op_data)

        except Exception as e:
            print(f"[警告] 获取算子 {cls.__name__} 的成本失败：{e}")

    return updated_pool

class OperatorExecutor:
    def __init__(self, strategy: str, t_limit: float, c_limit: float):
        """
        strategy: 'cost' or 'time'
        t_limit: total time limit
        c_limit: total resource (money) limit
        """
        self.strategy = strategy
        self.t_limit = t_limit
        self.c_limit = c_limit
        self.t_used = 0.0
        self.c_used = 0.0
        self.total_quality = 0.0
        self.execution_log = []

    # def calculate_cost(self, op: OperatorData) -> float:
    #     """Convert ci into actual cost in money."""
    #     if op.type == 'LLM':
    #         return op.ci * 0.0001
    #     elif op.type == 'CPU':
    #         return op.ci * 0.001
    #     else:
    #         raise ValueError(f"Unknown operator type: {op.type}")

    def compute_metrics(self, op: OperatorData) -> Dict[str, float]:
        Ti = op.ti * op.ri
        Qi = op.wi * op.ri
        Ci = op.ci
        Ri = Qi / Ci if self.strategy == 'cost' else Qi / Ti
        return {"Ti": Ti, "Qi": Qi, "Ci": Ci, "Ri": Ri}

    def choose_operator(self, operators: List[OperatorData]) -> Optional[OperatorData]:
        best_op = None
        best_metric = -float('inf')
        for op in operators:
            metrics = self.compute_metrics(op)
            if (self.t_used + metrics["Ti"] <= self.t_limit and
                self.c_used + metrics["Ci"] <= self.c_limit):
                if metrics["Ri"] > best_metric:
                    best_metric = metrics["Ri"]
                    best_op = (op, metrics)
        return best_op[0] if best_op else None


def test_operator_executor():
    operators = [
        OperatorData(name="CleanText", ti=0.3, ri=100, ci=200, wi=1.0, type="CPU"),
        OperatorData(name="SummarizeLLM", ti=0.5, ri=80, ci=10000, wi=1.8, type="LLM"),
        OperatorData(name="ParseLogs", ti=0.2, ri=120, ci=180, wi=0.9, type="CPU"),
        OperatorData(name="ClusterLLM", ti=0.6, ri=50, ci=8, wi=2.0, type="LLM"),
        OperatorData(name="Normalize", ti=0.1, ri=200, ci=100, wi=0.7, type="CPU")
    ]

    executor = OperatorExecutor(strategy="cost", t_limit=100, c_limit=1.0)

    print("=== 开始测试算子选择过程 ===\n")

    while True:
        op = executor.choose_operator(operators)
        if not op:
            print("\n💡 已达到时间/成本限制，结束选择。")
            break

        metrics = executor.compute_metrics(op)
        executor.t_used += metrics["Ti"]
        executor.c_used += metrics["Ci"]
        executor.total_quality += metrics["Qi"]
        executor.execution_log.append(op.name)

        print(f"✅ 选择算子：{op.name}")
        print(f"   → Ti={metrics['Ti']:.2f}, Ci={metrics['Ci']:.5f}, Qi={metrics['Qi']:.2f}, Ri={metrics['Ri']:.2f}")
        print(f"   → 累计时间：{executor.t_used:.2f} / {executor.t_limit}")
        print(f"   → 累计成本：{executor.c_used:.5f} / {executor.c_limit}\n")
        operators.remove(op)#删去该算子（也可不删）
    print("=== 最终执行日志 ===")
    print("选择顺序：", executor.execution_log)
    print("总时间使用：", round(executor.t_used, 2))
    print("总成本使用：", round(executor.c_used, 5))
    print("总质量收益：", round(executor.total_quality, 2))


# 运行测试
if __name__ == "__main__":
    test_operator_executor()


