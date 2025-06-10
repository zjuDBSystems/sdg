import subprocess
import os


def validate_js_syntax(file_path: str) -> bool:
    """
    验证JavaScript文件的语法正确性

    参数:
        file_path (str): 要验证的JavaScript文件路径

    返回:
        bool: 语法正确返回True，否则返回False
    """
    try:
        # 验证文件存在性
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        # 使用Node.js进行语法检查
        result = subprocess.run(
            ['node', '--check', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            text=True,
            encoding='utf-8'
        )

        # 输出错误信息便于调试
        # if result.returncode != 0:
        #     print(f"语法错误 ({file_path}):\n")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print(f"语法检查超时: {file_path}")
        return False
    except FileNotFoundError:
        print("Node.js 未安装，请先安装Node.js")
        return False
    except Exception as e:
        print(f"验证过程中发生异常 ({file_path}): {str(e)}")
        return False


def evaluate_js_folder(folder_path: str):
    """
    遍历文件夹，验证所有JS文件的语法，并计算得分

    参数:
        folder_path (str): JavaScript文件夹路径

    返回:
        float: 编译通过的得分（0-100）
    """
    # 获取文件夹中的所有文件
    js_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    total_files = len(js_files)
    # print(total_files)
    if total_files == 0:
        print("该文件夹没有JavaScript文件.")
        return 0.0

    # 验证每个文件的语法
    file_scores = {}
    passed_count = 0
    failed_files = []  # 新增：用于记录未能编译成功的文件列表

    for js_file in js_files:
        file_path = os.path.join(folder_path, js_file)
        # print(file_path)
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # 如果内容看起来像JSON格式，将其包裹在option =中
        if content.startswith("{") and content.endswith("}"):
            content = "option = " + content
        temp_file_path = os.path.join(folder_path, f"temp_{js_file}")
        # print(temp_file_path)
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            temp_file.write(content)
        # print(file_path)
        is_valid = validate_js_syntax(temp_file_path)
        # print(is_valid)
        # is_valid = validate_js_syntax(file_path)
        score = 100 if is_valid else 0
        file_scores[js_file] = score
        passed_count += 1 if is_valid else 0
        if not is_valid:
            failed_files.append(js_file)  # 将未能编译成功的文件添加到列表中
        # 删除临时文件
        os.remove(temp_file_path)

    score = (passed_count / total_files) * 100

    print("========== 语法检测指标评估结果 ==========")
    print(f"待检测的JavaScript文件总数: {total_files}")
    print(f"语法检测通过的文件数: {passed_count}")
    print(f"语法检测未通过的文件数: {total_files - passed_count}")
    print(f"语法检测得分（百分比）: {score:.2f}%")

    if score == 100:
        print("所有JavaScript文件语法均正确，说明该文件夹内的文件在语法层面上没有问题，质量较高。")
    elif score > 70:
        print("大部分JavaScript文件语法正确，但仍有部分文件存在语法错误，建议对未通过的文件进行检查和修正，以提升整体质量。未能编译成功的文件如下:")
        for file in failed_files:
            print(file)
    else:
        print("有较多JavaScript文件存在语法错误，这可能会影响到后续对这些文件的使用和处理，需要重点关注并修复这些语法问题。未能编译成功的文件如下:")
        for file in failed_files:
            print(file)

    return score, file_scores