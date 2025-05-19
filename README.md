# MultiArchKernelBench

A benchmark for evaluating LLMs' ability to generate kernels for various platform. Now supporting CUDA kernels for GPUs and Ascendc kernels for NPUs.

# Usage
## CUDA kernels
#### Config
Set configurations in config.py, including op and project_root_path.
#### run
```
python mainloop_cuda.py
```

## Ascendc kernels
#### Config
Set configurations in config.py, including op and project_root_path.

#### run
```
python mainloop_ascendc.py
```

## Support for New Operators

### 1. Add New Operators to the Dataset
Update the `dataset` dictionary in `constant.py` by adding entries for the new Python operator items.

### 2. Add Reference Implementation for Correctness Verification
- Create a new file in the `reference` directory. The file name **must exactly match** the kernel name specified in the `dataset`.
- Implement the reference function using official PyTorch APIs. The function should be named as:
  
  ```python
  {kernel_name}_reference
  ```
- Define a `get_inputs` function that generates input tensors consistent with those in the `dataset`.

### Add AscendC Operator Projects (Optional)
- Create a JSON file in the `ascend_op_projects` directory, and then use msopgen to create project automatically.
- The operator name in the JSON file must use CamelCase format based on the kernel name (e.g., `elementwise_add` → `ElementwiseAdd`).
- When using msopgen to generate the project, ensure the project name specified with the --out option also uses CamelCase format.
