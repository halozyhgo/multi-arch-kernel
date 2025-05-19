import os
import subprocess
from config import op_engineer_dir, deploy_path

def underscore_to_pascalcase(underscore_str):
    """
    Convert underscore-separated string to PascalCase.
    
    Args:
        underscore_str (str): Input string with underscores (e.g., "vector_add")
        
    Returns:
        str: PascalCase version (e.g., "VectorAdd")
    """
    if not underscore_str:  # Handle empty string
        return ""
    
    parts = underscore_str.split('_')
    # Capitalize the first letter of each part and join
    return ''.join(word.capitalize() for word in parts if word)


def ascend_compile(generated_code, op):

    op_capital=underscore_to_pascalcase(op)
    target_directory=os.path.join(op_engineer_dir, op_capital)
    
    try:
        context={}
        compile(generated_code, "<string>", "exec")
        exec(generated_code, context)  # For Python, use exec() (be careful with untrusted code)
    except Exception as e:
        raise Exception(f'Error in generated code {e}')

    # write code to specific location
    with open(os.path.join(target_directory, 'op_host', f'{op}_tiling.h'), 'w') as f:
        f.write(context.get('host_tiling_src'))

    with open(os.path.join(target_directory, 'op_host', f'{op}.cpp'), 'w') as f:
        f.write(context.get('host_operator_src'))

    with open(os.path.join(target_directory, 'op_kernel', f'{op}.cpp'), 'w') as f:
        f.write(context.get('kernel_src'))

    with open(os.path.join(op_engineer_dir, 'CppExtension', 'csrc', f'op.cpp'), 'w') as f:
        f.write(context.get('python_bind_src'))

    try:
        print("Begin build")
        os.chdir(target_directory)
        result = subprocess.run(["./build.sh"], check=True, capture_output=True, text=True)
        print("Build succeeded")
    except subprocess.CalledProcessError as e:
        print("Build failed!")
        # print("Exit Code:", e.returncode)
        # print("Error Output:\n", e.stdout)
        # print("Error Output:\n", e.stderr)
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)



    try:
        print("Begin deploy")
        os.chdir(os.path.join(target_directory, 'build_out'))
        result = subprocess.run(["./custom_opp_ubuntu_aarch64.run", f'--install-path={deploy_path}'], check=True, capture_output=True, text=True)
        print("Deploy succeeded")
    except subprocess.CalledProcessError as e:
        print("Deploy failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)



    try:
        print("Begin pybind")
        os.chdir(os.path.join(op_engineer_dir, 'CppExtension'))
        result = subprocess.run(['bash', "build_and_run.sh"], check=True, capture_output=True, text=True)
        print("Pybind succeeded\n")
    except subprocess.CalledProcessError as e:
        # Print error if build.sh fails
        print("Pybind failed!")
        feedback = f'Exit Code: {e.returncode}\nError Output:\n{e.stdout}'
        raise Exception(feedback)

if __name__ == '__main__':
    import torch
    import torch_npu
    import custom_ops_lib
    op = 'relu'
    generated_method = getattr(custom_ops_lib, op)