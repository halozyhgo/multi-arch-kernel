from get_prompt import generate_one_prompt, generate_one_feedback_prompt
from config import *
import os
from openai import OpenAI
import re
import torch
import ast
import subprocess
import csv
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.profiler as xp
xp.start_server(9012)
import torch_xla.debug.metrics as met


# 获取单个 TPU 核心
device = xm.xla_device()

# 调整运行核函数的代码模板，使用 XLA 设备
call_kernel_code = '''
with torch.no_grad():
    for trial in range({}):
        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        output = {}(*inputs)
'''


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
    return os.path.join(ref_impl_base_path, f'{op}.py')


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

def profile_op(get_inputs, method, method_name):
    '''
    :param get_inputs:获取输入数据 
    :param method: 执行的方法
    :param method_name: 执行的方法名称
    :return: 执行时间
    '''
    met.clear_counters()
    for i in range(10):
        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        with xp.StepTrace(f"{method_name}_{i}"):
            output = method(*inputs)
    res = met.short_metrics_report()
    lines = res.splitlines()
    result = {}
    current_key = None
    current_dict = None
    for line in lines:
        line = line.strip()
        if line.startswith("Counter:"):
            current_key = line.split(":")[1].strip()
            result[current_key] = {}
            current_dict = result[current_key]
        elif line.startswith("Metric:"):
            current_key = line.split(":")[1].strip()
            result[current_key] = {}
            current_dict = result[current_key]
        elif line:
            sub_key, sub_value = line.split(":", 1)
            sub_key = sub_key.strip()
            sub_value = sub_value.strip()
            current_dict[sub_key] = sub_value

    time_cosum = result['ExecuteTime']['Accumulator']
    return time_cosum

def main():
    prompt = generate_one_prompt('pallas', op)
    ref_src_path = get_ref_src_path(op)

    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
        timeout=10000000,
        max_retries=3,
    )

    cur_turn = 0
    while cur_turn < max_turn:
        cur_turn += 1
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=temperature,
            n=num_completions,
            max_tokens=max_tokens,
            top_p=top_p
        )
        outputs = [choice.message.content for choice in response.choices]
        assert len(outputs) == 1
        output = outputs[0]
        generated_code, code_block = extract_first_code(output, ["python"])
        if generated_code is None:
            print('Generated code does not in code blocks')
            break

        # compile
        try:
            context = {}
            compile(generated_code, "<string>", "exec")
            exec(generated_code, context)  # For Python, use exec() (be careful with untrusted code)
            print('Compiling successful!!')
        except Exception as e:
            print('Compiling fail!!')
            error_message = str(e)
            prompt = generate_one_feedback_prompt('cuda', op, code_block, error_message)
            continue

        # correctness
        with open(ref_src_path, 'r') as f:
            ref_src = f.read()
        exec(ref_src, context)
        get_inputs = context['get_inputs']
        ref_method = context[f'{op}_reference']
        generated_method = context[op]

        with torch.no_grad():
            for trial in range(num_correct_trials):
                inputs = get_inputs()
                inputs = [
                    x.to(device) if isinstance(x, torch.Tensor) else x
                    for x in inputs
                ]
                ref_output = ref_method(*inputs)
                new_output = generated_method(*inputs)
                feedback = None
                if ref_output.shape != new_output.shape:
                    feedback = f"Output shape mismatch: Expected {ref_output.shape}, got {new_output.shape}"
                elif not torch.allclose(ref_output, new_output, atol=1e-02, rtol=1e-02):
                    feedback = f"[FAIL] Output mismatch"
                
                if feedback is None:
                    print('Correct Implementation!!')
                else:
                    print('Wrong Implementation!!')
                    prompt = generate_one_feedback_prompt('cuda', op, code_block, feedback)
                    continue

        # get ExecuteTime
        ExecuteTime_gen = profile_op(get_inputs,generated_method,'geneated')
        ExecuteTime_ref = profile_op(get_inputs,ref_method,'geneated')

        pd.DataFrame({'ExecuteTime_gen':ExecuteTime_gen,'ExecuteTime_ref':ExecuteTime_ref},index=[0]).to_csv(f'{op}_pallas_ExecuteTime.csv')        
        prompt = generate_one_feedback_prompt('cuda', op, code_block, profiler_feedback)
        print(prompt)
        break


if __name__ == '__main__':
    main()