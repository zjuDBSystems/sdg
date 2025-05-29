"""PoC code to demonstrate the usage of the data_operator package.
"""

import os
import json
import time

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
                api_key = "sk-dC9449cf83366aa25e16e59cf7fa08192a79497025fKY2m9"
            ),
            # 图像的echarts代码补全
            registry['ImgToEchartsOperator'](
                api_key = "sk-dC9449cf83366aa25e16e59cf7fa08192a79497025fKY2m9"
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
