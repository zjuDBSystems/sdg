import os

from .storage.dataset import Dataset, DataType, Datadir, copy_dataset

from .event import global_message_queue, EventType, EventResponse
from .data_operator.operator import OperatorMeta
from .cost_evaluation import OperatorData,OperatorExecutor,refresh_operator_costs,flatten_secondary_metrics,operator_to_metrics
from .task.task import Task

from .data_insights_identify import sort_metrics
from openai import OpenAI
from datetime import datetime
import json
# LOG_FILE_PATH = "./operator_selection_log.jsonl"
def generate_negative_correlation_weights(result: dict) -> dict:
    """
    根据质量评估结果生成负相关权重。
    分数越低，权重越高，并归一化到 0-100 范围。
    """
    import numpy as np

    # 获取所有得分并反向处理
    scores = np.array(list(result.values()), dtype=np.float64)
    reversed_scores = 100 - scores  # 分数低 => 差距大 => 权重高

    # 归一化到 0-100
    min_val = reversed_scores.min()
    max_val = reversed_scores.max()
    if max_val == min_val:
        normalized_weights = np.ones_like(reversed_scores) * 50  # 所有值相等时统一设为50
    else:
        normalized_weights = 100 * (reversed_scores - min_val) / (max_val - min_val)

    # 生成字典形式
    sorted_total_weights = {
        key: float(f"{weight:.4f}")  # 保留小数点后 4 位
        for key, weight in zip(result.keys(), normalized_weights)
    }

    # 按权重降序排列（可选）
    sorted_total_weights = dict(
        sorted(sorted_total_weights.items(), key=lambda item: item[1], reverse=True)
    )

    return sorted_total_weights

# 固定住时间戳，只生成一次
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = f"operator_selection_log_{timestamp_str}.json"
def log_iteration(sorted_total_weights, selected_operator, result, is_initial=False):
    # timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # log_path = f"operator_selection_log_{timestamp_str}.json"

    log_entry = {
        "timestamp": datetime.now().isoformat(),
    }

    if is_initial:
        log_entry["initial_result"] = result
    else:
        log_entry["selected_operator"] = selected_operator
        log_entry["result"] = result
        log_entry["sorted_total_weights"] = sorted_total_weights

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")  # 分隔线，便于人工查看
        f.write(json.dumps(log_entry, indent=2, ensure_ascii=False))  # 中文 & 缩进
        f.write("\n")


def describe_data(datadir: Datadir):
    dir_path = datadir.data_path
    count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
    data_type = datadir.data_type.value
    global_message_queue.put(EventResponse(EventType.REASONING, f'{data_type} data in {dir_path} has {count} files!'))


def describe_metadata(metadata_path: str):
    with open(metadata_path, 'r') as f:
        lines = f.readlines()
    global_message_queue.put(
        EventResponse(EventType.REASONING, f'multimodal dataset contains {len(lines) - 1} data pairs!'))


def test_cost():
    #构建数据集
    registry = OperatorMeta.get_registry()
    # print(registry)

    code_dir = Datadir('echart-code-sample-negative', DataType.CODE)

    describe_data(code_dir)
    image_dir = Datadir('echart-image-sample-negative', DataType.IMAGE)
    describe_data(image_dir)
    data_set = Dataset([code_dir, image_dir], 'echart-sample-negative.metadata', 'key_configurations.md')


    #初次数据质量评估
    result = data_set.evaluate_image_code_quality()
    print(result)
    # 💾 记录初始质量评估
    log_iteration(None, None, result, is_initial=True)
    #将质量评估结果转换成靶点发现期望的形式
    result=flatten_secondary_metrics(result)
    # print(result)

    #进行靶点发现，获取靶点权重
    client = OpenAI(api_key="sk-3955d8823efd4f2483897446b91a7ffb", base_url="https://api.deepseek.com")
    sorted_metrics, sorted_total_weights = sort_metrics(client=client, code_quality_analysis=result,
                                                        llm_weight=0.7)
    # sorted_total_weights=generate_negative_correlation_weights(result)
    # print(sorted_metrics)
    # print(sorted_total_weights)
    # exit(0)

    # 第一次构建可选算子池
    operator_pool = []
    # 遍历注册表中所有算子
    for cls in registry.values():
        instance = cls()
        cost_info = instance.get_cost(data_set)
        name = cost_info["name"]

        # 默认权重为 0
        weight = 0.0

        # 查找该算子对应的指标
        metric_list = operator_to_metrics.get(name, [])

        # 累加该算子对应指标的权重
        for metric in metric_list:
            weight += sorted_total_weights.get(metric, 0.0)

        # 构建 OperatorData 对象
        op_data = OperatorData(
            name=name,
            ti=cost_info["ti"],
            ri=cost_info["ri"],
            ci=cost_info["ci"],
            wi=weight,
            type=cost_info["type"]
        )
        operator_pool.append(op_data)
    print(operator_pool)


    #初始化代价评估器
    executor = OperatorExecutor(strategy='cost', t_limit=100, c_limit=10)
    while True:
        #选中下一个最应该执行的算子
        op = executor.choose_operator(operator_pool)
        if not op:
            print("\n💡 已达到时间/成本限制，结束选择。")
            break
        #更新代价评估器
        metrics = executor.compute_metrics(op)
        executor.t_used += metrics["Ti"]
        executor.c_used += metrics["Ci"]
        executor.total_quality += metrics["Qi"]
        executor.execution_log.append(op.name)

        print(f"✅ 选择算子：{op.name}")
        print(f"   → Ti={metrics['Ti']:.2f}, Ci={metrics['Ci']:.5f}, Qi={metrics['Qi']:.2f}, Ri={metrics['Ri']:.2f}")
        print(f"   → 累计时间：{executor.t_used:.2f} / {executor.t_limit}")
        print(f"   → 累计成本：{executor.c_used:.5f} / {executor.c_limit}\n")


        # 执行对应的算子，我这里调试有些困难
        task = Task(
            [
                # 配置项修正
                registry[op.name](),

            ],
            data_set
        )
        result=task.run()
        print(result)
        # print("结果1")
        data_set = task.final_dataset
        # operator_pool.remove(op)#删去该算子（也可不删），删去表示执行过的不会被再次执行，不删则表示可能后续还会再次调用该算子

        #重新进行质量评估并更新算子库的各个参数
        # result = data_set.evaluate_image_code_quality()
        # print(result)
        # print("结果2")
        # exit(0)

        result = flatten_secondary_metrics(result)
        sorted_metrics, sorted_total_weights = sort_metrics(client=client, code_quality_analysis=result,
                                                            llm_weight=0.7)
        # sorted_total_weights = generate_negative_correlation_weights(result)
        # 💾 记录本轮选择的指标与算子
        log_iteration(sorted_total_weights, op.name, result)#用于记录
        operator_pool=refresh_operator_costs(registry, operator_pool, data_set,sorted_total_weights=sorted_total_weights,operator_to_metrics=operator_to_metrics)
        print(operator_pool)

    # exit(0)



if __name__ == '__main__':
    test_cost()