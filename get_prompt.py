
from constant import ascendc_instruction, add_cuda_example, add_ascendc_pipline, cuda_instruction,pallas_instruction, dataset


language2instruction = {'cuda': cuda_instruction, 'Ascendc': ascendc_instruction,'pallas':pallas_instruction}
language2example = {'cuda': add_cuda_example, 'Ascendc': add_ascendc_pipline}
language2block = {'cuda': 'python', 'Ascendc': 'cpp','pallas':'python'}
language2generated = {'cuda': 'Code', 'Ascendc': 'Ascendc Kernel','pallas': 'Code'}


def one_shot(language: str):
    prompt = f"""
### Example:
#### Input Tensors:
- x: shape (2, 2048), dtype float32
- y: shape (2, 2048), dtype float32
#### Output Tensors:
- z: shape (2, 2048), dtype float32
#### Kernel Name: `elementwise_add`
#### Operation: `z = x + y`
#### Generated {language2generated[language]}:
```{language2block[language]}
{language2example[language]}
```
    """
    return prompt


def one_shot_prompt_generate_custom(
    language:str, kernel_name:str, input_tensors:list, output_tensors:list, operation:str
) -> str:
    input_str = ''
    for tensor in input_tensors:
        input_str += f'- {tensor}\n'
    output_str = ''
    for tensor in output_tensors:
        output_str += f'- {tensor}\n'
    prompt = f"""
### Task Description:
{language2instruction[language]}
{one_shot(language)}
### New Task:
#### Input Tensors:
{input_str}
#### Output Tensors:
{output_str}
#### Kernel Name: `{kernel_name}`
#### Operation: `{operation}`
#### Generated {language2generated[language]}:
    """
    return prompt

def generate_feedback_prompt(language:str, kernel_name:str, input_tensors:list, operation:str, previous_code_block:str, feedback:str):
    input_str = ''
    for tensor in input_tensors:
        input_str += f'- {tensor}\n'
    prompt = f"""
### Task Description:
{language2instruction[language]}
#### Input Tensors:
{input_str}
#### Kernel Name: `{kernel_name}`
#### Operation: `{operation}`

#### Your Latest Generated Code::
{previous_code_block}

#### Feedback:
{feedback}
#### Please provide an improved version of the code:
"""
    return prompt

def generate_one_feedback_prompt(language:str, kernel_name:str, previous_code, feedback):
    info = dataset[kernel_name]
    return generate_feedback_prompt(language, kernel_name, info['Input Tensors'], info['operation'], previous_code, feedback)

def generate_one_prompt(language:str, kernel_name:str):
    info = dataset[kernel_name]
    return one_shot_prompt_generate_custom(language, kernel_name, info['Input Tensors'], info['Output Tensors'], info['operation'])

if __name__ == '__main__':
    print(generate_one_prompt('cuda','elementwise_sub'))
    # print(generate_one_feedback_prompt('cuda','softmax', 'sd', 'feedback'))