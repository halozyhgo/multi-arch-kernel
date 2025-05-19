from get_prompt import generate_one_prompt, generate_one_feedback_prompt
import config
import os
from openai import OpenAI
import re
import torch
# import torch_npu
import ast
from ascend_compile_pipeline import ascend_compile
from utils.source_code_process import *
call_kernel_code='''
with torch.no_grad():
    for trial in range({}):
        inputs = get_inputs()
        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        output = {}(*inputs)
'''

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

def main():
    # op = config.op
    # op = 'elementwise_sub'
    op_url_list,op_name_list = get_op_name_list('./reference/level1')


    prompt = generate_one_prompt('Ascendc',op)
    ref_src_path = get_ref_src_path(op)
    DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
    client = OpenAI(
        api_key=DEEPSEEK_KEY,
        base_url="https://api.deepseek.com",
        timeout=10000000,
        max_retries=3,
    )    
    cur_turn = 0

    while cur_turn < config.max_turn:
        cur_turn += 1
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=config.temperature,
            n=config.num_completions,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )

        outputs = [choice.message.content for choice in response.choices]
        assert len(outputs) == 1
        output = outputs[0]
        generated_code, code_block = extract_first_code(output, ["python", "cpp"])

        if generated_code is None:
            print('Generated code does not in code blocks')
            break    

        
        # compile
        try:
            ascend_compile(generated_code, op)
        except Exception as e:
            error_message = str(e)
            print(error_message)
            prompt = generate_one_feedback_prompt('cuda', op, code_block, error_message)
            continue

        # correctness
        context = {}
        with open(ref_src_path, 'r') as f:
            ref_src = f.read()
        exec(ref_src, context)
        get_inputs = context['get_inputs']
        ref_method = context[f'{op}_reference']

        if 'ASCEND_CUSTOM_OPP_PATH' not in os.environ:
            raise Exception('Please first run source opp/vendors/customize/bin/set_env.sh in operator deploy path')

        import custom_ops_lib
        generated_method = getattr(custom_ops_lib, op)
        num_correct_trials = 5
        with torch.no_grad():
            for trial in range(num_correct_trials):
                inputs = get_inputs()
                inputs = [
                    x.npu() if isinstance(x, torch.Tensor) else x
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
                    prompt = generate_one_feedback_prompt('cuda', op, code_block, feedback)
                    continue                    
                    



        # # get profiling data
        # inputs_code = get_function_source(ref_src_path, 'get_inputs')     
        # cuda_code_file_path = 'tmp.py'
        # profiler_out = 'tmp.csv'
        # trail=5
        # profiler_advice = {}
        # profiler_feedback = ''

        # with open(cuda_code_file_path, 'w') as cuda_code_file:
        #     cuda_code_file.write(generated_code + '\n' + inputs_code + '\n\n'  + call_kernel_code.format(trail,op))
        # result = subprocess.run(["ncu", "--csv", "--log-file", profiler_out, "python", cuda_code_file_path])  # Linux/macOS
        # with open(profiler_out, 'r') as file:
        #     lines = file.readlines()
        # # Filter out lines starting with '==PROF=='
        # cleaned_lines = [line for line in lines if not line.startswith("==PROF==")]
        # # Write cleaned data to a new file
        # with open(profiler_out, 'w') as file:
        #     file.writelines(cleaned_lines)

        # with open(profiler_out, mode='r', encoding='utf-8') as file:
        #     reader = csv.DictReader(file)
        #     for row in reader:
        #         if row['Rule Type'] == 'OPT':
        #             profiler_advice[(row['Kernel Name'], row['Rule Name'])] = row['Rule Description']
        #             profiler_feedback += f'kernel name: {row["Kernel Name"]}\nprofiler advice: {row["Rule Description"]}\n\n'
        # prompt = generate_one_feedback_prompt('cuda', op, code_block, profiler_feedback)
        # print(prompt)
        break
        

if __name__ == '__main__':
    main()
