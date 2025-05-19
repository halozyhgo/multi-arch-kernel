
# op
op = 'elementwise_sub'

# path
project_root_path = '/root/autodl-tmp/multi-arch-kernel-bench/'
ref_impl_base_path = f'{project_root_path}/reference/level1'

# trial
max_turn = 1
num_correct_trials = 5

# LLM config
max_tokens = 4096
temperature = 0.0
top_p=1.0
num_completions=1


# Ascend compile related
op_engineer_dir = f'{project_root_path}/ascend_op_projects'
deploy_path = f'{op_engineer_dir}/opp'


cuda_call_kernel_code='''
with torch.no_grad():
    for trial in range({}):
        inputs = get_inputs()
        inputs = [
            x.cuda() if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        output = {}(*inputs)
'''

pallas_call_kernel_code = '''
with torch.no_grad():
    for trial in range({}):
        inputs = get_inputs()
        inputs = [
            x.to(device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        output = {}(*inputs)
'''