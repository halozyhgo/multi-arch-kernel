from get_prompt import generate_one_prompt, generate_one_feedback_prompt
from config import *
import os
from openai import OpenAI
import re
import torch
import ast
import subprocess
import csv
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



def main():
    prompt = generate_one_prompt('cuda',op)
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
        generated_code, code_block = extract_first_code(output, ["python", "cpp"])

        if generated_code is None:
            print('Generated code does not in code blocks')
            break    

        
        # compile
        try:
            context={}
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
                    x.cuda() if isinstance(x, torch.Tensor) else x
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
                    



        # get profiling data
        inputs_code = get_function_source(ref_src_path, 'get_inputs')     
        cuda_code_file_path = 'tmp.py'
        profiler_out = 'tmp.csv'
        trail=5
        profiler_advice = {}
        profiler_feedback = ''

        with open(cuda_code_file_path, 'w') as cuda_code_file:
            cuda_code_file.write(generated_code + '\n' + inputs_code + '\n\n'  + call_kernel_code.format(trail,op))
        result = subprocess.run(["ncu", "--csv", "--log-file", profiler_out, "python", cuda_code_file_path])  # Linux/macOS
        with open(profiler_out, 'r') as file:
            lines = file.readlines()
        # Filter out lines starting with '==PROF=='
        cleaned_lines = [line for line in lines if not line.startswith("==PROF==")]
        # Write cleaned data to a new file
        with open(profiler_out, 'w') as file:
            file.writelines(cleaned_lines)

        with open(profiler_out, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['Rule Type'] == 'OPT':
                    profiler_advice[(row['Kernel Name'], row['Rule Name'])] = row['Rule Description']
                    profiler_feedback += f'kernel name: {row["Kernel Name"]}\nprofiler advice: {row["Rule Description"]}\n\n'
        prompt = generate_one_feedback_prompt('cuda', op, code_block, profiler_feedback)
        print(prompt)
        break
        

if __name__ == '__main__':
    main()
