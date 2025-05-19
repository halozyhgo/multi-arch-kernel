from get_prompt import generate_one_prompt, generate_one_feedback_prompt
from config import *
import os
from openai import OpenAI
import re
import torch
import ast
import subprocess
import csv

def extract_first_code(output_string: str, code_language_types: list[str]) -> str:
    """
    Extract first code block from model output, specified by code_language_type
    """
    trimmed = output_string.strip()

    # Extracting the first occurrence of content between backticks
    code_match = re.search(r"```(.*?)```", trimmed, re.DOTALL)

    if code_match:
        # Strip leading and trailing whitespace from the extracted code
        code_block = code_match.group(1).strip()

        # depends on code_language_type: cpp, python, etc.
        # sometimes the block of code is ```cpp ... ``` instead of ``` ... ```
        # in this case strip the cpp out
        for code_type in code_language_types:
            if code_block.startswith(code_type):
                code = code_block[len(code_type):].strip()

        return code, f'```{code_block}```'

    return None, None


def get_ref_src_path(op):
    return os.path.join(ref_impl_base_path, f'{op}')


def get_two_init_func_src(ref_code):
    return get_function_source(ref_code, 'get_init_inputs'), get_function_source(ref_code, 'get_inputs')


def get_function_source(file_path, func_name):
    """Extracts the source code of a function from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        source_code = f.read()

    tree = ast.parse(source_code)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            start_line = node.lineno - 1  # Convert 1-based to 0-based index
            end_line = max([n.lineno for n in ast.walk(node) if hasattr(n, "lineno")])  # Get last line
            return "\n".join(source_code.splitlines()[start_line:end_line])

    raise ValueError(f"Function '{func_name}' not found in {file_path}")

def get_op_name_list(path):
    op_url_list = os.listdir(path)
    target_name_list = []
    # todo:重命名文件名
    for i in range(len(op_url_list)):
        old_name = os.path.join(path,op_url_list[i])
        target_name = remove_number_and_extra_underscores(op_url_list[i])
        target_name_list.append(target_name)
        target_name = os.path.join(path,target_name)
        os.rename(old_name, target_name)
    return target_name_list


def remove_number_and_extra_underscores(old_name):
    # 分离文件名和扩展名
    base_name, ext = os.path.splitext(old_name)

    # 移除开头的数字和下划线
    new_base = re.sub(r'^\d+_', '', base_name)

    # 移除末尾的下划线
    new_base = new_base.rstrip('_')

    return new_base + ext


if __name__ == '__main__':
    op_name_list = get_op_name_list('../reference/level1')
    print(op_name_list)

    # ori_name = '1_Square_matrix_multiplication_.py'
    # target_name = remove_number_and_extra_underscores(ori_name)
    # print(target_name)

