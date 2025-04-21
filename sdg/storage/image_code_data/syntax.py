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
            text=True
        )

        # 输出错误信息便于调试
        if result.returncode != 0:
            print(f"语法错误 ({file_path}):\n{result.stderr.strip()}")

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
def evaluate_js_folder(folder_path: str) :
    """
    遍历文件夹，验证所有JS文件的语法，并计算得分

    参数:
        folder_path (str): JavaScript文件夹路径

    返回:
        float: 编译通过的得分（0-100）
    """
    # 获取文件夹中的所有文件
    js_files = [f for f in os.listdir(folder_path) if f.endswith(".js")]

    total_files = len(js_files)
    if total_files == 0:
        print("该文件夹没有JavaScript文件.")
        return 0.0

    # 验证每个文件的语法
    file_scores = {}
    passed_count = 0

    for js_file in js_files:
        file_path = os.path.join(folder_path, js_file)
        is_valid = validate_js_syntax(file_path)
        score = 100 if is_valid else 0
        file_scores[js_file] = score
        passed_count += 1 if is_valid else 0

    score = (passed_count / total_files) * 100


    return score,file_scores

