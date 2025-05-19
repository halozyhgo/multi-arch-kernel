ascendc_instruction = 'You are an expert Ascendc kernel developer. Given input tensors, a kernel name, and an operation, generate an optimized Ascendc kernel. Your output should include four Python strings: `host_tiling_src`, `host_operator_src`, `kernel_src`, and `python_bind_src`. The kernel function name must exactly match the provided kernel name. The operator definition in the host code should also correspond to the kernel name, but follow PascalCase naming.. Follow the example below.'
cuda_instruction = 'Your task is to generate an optimized CUDA kernel based on the given input and output tensors, kernel name, and the operation. Then, compile and load the kernel as a PyTorch C++ extension, and wrap it in a Python function with the same name as the gien kernel. Follow the example below.'
cuda_feedback_instruction = 'You are an expert CUDA kernel developer. Your task is to generate an optimized CUDA kernel based on the provided input tensors, a specified kernel name, and a given mathematical operation. After generating the source code, you will compile and load it as a PyTorch C++ extension and integrate it into a PyTorch module. The resulting model should be named ModelNew. You have already generated the code—now, please refine and improve it based on the feedback.'

# pallas_instruction = 'Your task is to generate an optimized Pallas kernel based on the given input and output tensors, kernel name, and the operation. Then, compile and load the kernel as a PyTorch C++ extension, and wrap it in a Python function with the same name as the gien kernel. Follow the example below.'
pallas_instruction = 'Your task is to generate a optimized kernel using JAX/Pallas based on the given input and output tensors, kernel name, and operation. Then, compile and load the kernel through the torch_xla interface, and wrap it in a Python function matching the given kernel name. Follow the example below.'

add_ascendc_pipline='''
host_tiling_src="""
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ElementwiseAddTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ElementwiseAdd, ElementwiseAddTilingData)
}
"""

host_operator_src="""
#include "elementwise_add_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    ElementwiseAddTilingData tiling;
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(TILE_NUM);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
const auto inputDataType = context->GetInputDataType(0);
context->SetOutputDataType(0, inputDataType);
return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class ElementwiseAdd : public OpDef {
public:
    explicit ElementwiseAdd(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(ElementwiseAdd);
}
"""

kernel_src="""
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void elementwise_add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
"""

python_bind_src="""
#include <torch/library.h>
#include <torch/csrc/autograd/custom_function.h>
#include "pytorch_npu_helper.hpp"
#include <torch/extension.h>

// 为NPU设备注册前向实现
at::Tensor elementwise_add_impl_npu(const at::Tensor& self, const at::Tensor& other) {
    // 创建输出内存
    at::Tensor result = at::empty_like(self);

    // 调用aclnn接口计算
    EXEC_NPU_CMD(aclnnElementwiseAdd, self, other, result);
    return result;
}


// 为NPU设备注册前反向实现
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
    m.impl("elementwise_add", &elementwise_add_impl_npu);
}

// // 通过pybind将c++接口和python接口绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("elementwise_add", &elementwise_add_impl_npu, "x + y");
}
"""


'''

add_ascend_example='''
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void my_add_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}

void vector_add(void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
{
    int blockDim = 8;
    my_add_kernel<<<blockDim, nullptr, stream>>>(x, y, z);
}

'''

add_cuda_example='''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for element-wise addition
elementwise_add_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void elementwise_add_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    elementwise_add_kernel<<<num_blocks, block_size>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), size);

    return out;
}
"""

elementwise_add_cpp_source = (
    "torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);"
)

# Compile the inline CUDA code for element-wise addition
elementwise_add_module = load_inline(
    name="elementwise_add",
    cpp_sources=elementwise_add_cpp_source,
    cuda_sources=elementwise_add_source,
    functions=["elementwise_add_cuda"],
    verbose=True,
    extra_cflags=[""],
    extra_ldflags=[""],
)

def vector_add(x, y):
    return elementwise_add_module.elementwise_add_cuda(x, y)
'''

add_pallas_example = '''
from torch_xla.experimental.custom_kernel import jax_import_guard
jax_import_guard()

import jax
from jax.experimental import pallas as pl
import jax.numpy as jnp

# Define the custom Pallas kernel for element-wise addition
def elementwise_add_kernel(x_ref, y_ref, z_ref):
    x = x_ref[...]  # Load full tensor using memory reference
    y = y_ref[...]
    z_ref[...] = x + y  # Store result

@jax.jit
def elementwise_add_pallas(x: jax.Array, y: jax.Array) -> jax.Array:
    return pl.pallas_call(
        elementwise_add_kernel,
        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype)
    )(x, y)

# Compile and create PyTorch compatible kernel
from torch_xla.experimental.custom_kernel import make_kernel_from_pallas
elementwise_add_module = make_kernel_from_pallas(
    elementwise_add_pallas, 
    lambda x, y: [(x.shape, x.dtype)]  # Shape inference function
)

def elementwise_add(x, y):
    return elementwise_add_module(x, y)
'''

dataset = {
    # math
    'elementwise_sub': {
        'Input Tensors': [
            'x: shape (2, 2048), dtype float32',
            'y: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'z: shape (2, 2048), dtype float32'
        ],
        'operation': 'z = x - y',
        'category': 'math'
    },
    'elementwise_div': {
        'Input Tensors': [
            'x: shape (2, 2048), dtype float32',
            'y: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'z: shape (2, 2048), dtype float32'
        ],
        'operation': 'z = x / y',
        'category': 'math'
    },

    'elementwise_mul': {
        'Input Tensors': [
            'x: shape (2, 2048), dtype float32',
            'y: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'z: shape (2, 2048), dtype float32'
        ],
        'operation': 'z = x * y',
        'category': 'math'
    },

    'elementwise_pow': {
        'Input Tensors': [
            'x: shape (2, 2048), dtype float32',
            'y: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'z: shape (2, 2048), dtype float32'
        ],
        'operation': 'z = x ** y',
        'category': 'math'
    },

    'abs': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = abs(input)',
        'category': 'math'
    },

    'exp': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = exp(input)',
        'category': 'math'
    },

    'clamp': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32',
            'min_val: shape (2, 2048), dtype float32',
            'max_val: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = min(max(input, min_val), max_val)',
        'category': 'math'
    },

    'reciprocal': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = 1 / input',
        'category': 'math'
    },

    'rsqrt': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = 1 / sqrt(input)',
        'category': 'math'
    },

    'neg': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = -input',
        'category': 'math'
    },

    'cos': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = cos(input)',
        'category': 'math'
    },

    'sin': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = sin(input)',
        'category': 'math'
    },

    'tanh': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = tanh(input)',
        'category': 'math'
    },
    
    # activation
    'relu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = max(0, input)',
        'category': 'activation'
    },
    'leaky_relu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = input if input > 0 else 0.01 * input',
        'category': 'activation'
    },
    'sigmoid': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = 1 / (1 + exp(-input))',
        'category': 'activation'
    },

    'softmax': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = exp(input) / sum(exp(input), axis=-1, keepdims=True)',
        'category': 'activation'
    },

    'gelu': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = 0.5 * input * (1 + tanh(√(2/π) * (input + 0.044715 * input^3)))',
        'category': 'activation'
    },

    'silu': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = input / (1 + exp(-input))',
        'category': 'activation'
    },

    'log_softmax': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = log(softmax(input)) = input - log(sum(exp(input), axis=-1, keepdims=True))',
        'category': 'activation'
    },


    'matrix_multiplication': {
        'Input Tensors': [
            'a: shape (1024, 4096), dtype float32',
            'b: shape (4096, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (1024, 2048), dtype float32'
        ],
        'operation': 'output = matmul(a, b)',
        'category': 'blas'
    },

    'batch_matrix_multiplication': {
        'Input Tensors': [
            'a: shape (128, 128, 256), dtype float32',
            'b: shape (128, 256, 128), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (128, 128), dtype float32'
        ],
        'operation': 'output = batched_matmul(a, b)',
        'category': 'blas' 
    },
    'matrix_vector_multiplication': {
        'Input Tensors': [
            'matrix: shape (16, 2048), dtype float32',
            'vector: shape (2048, 1), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 1), dtype float32'
        ],
        'operation': 'output = matmul(matrix, vector)',
        'category': 'blas'
    },

    'outer': {
        'Input Tensors': [
            'x: shape (1024,1), dtype float32',
            'y: shape (1024,1), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (1024, 1024), dtype float32'
        ],
        'operation': 'output[i][j] = x[i] * y[j]',
        'category': 'blas'
    },

    'sum': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 1), dtype float32'
        ],
        'operation': 'output = sum(input, axis=-1, keepdims=True)',
        'category': 'reduction'
    },

    'mean': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 1), dtype float32'
        ],
        'operation': 'output = mean(input, axis=-1, keepdims=True)',
        'category': 'reduction'
    },

    'prod': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 1), dtype float32'
        ],
        'operation': 'output = prod(input, axis=-1, keepdims=True)',
        'category': 'reduction'
    },

    'max': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 1), dtype float32'
        ],
        'operation': 'output = max(input, axis=-1, keepdims=True)',
        'category': 'reduction'
    },

    'min': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 1), dtype float32'
        ],
        'operation': 'output = min(input, axis=-1, keepdims=True)',
        'category': 'reduction'
    },

    'argmax': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'indices: shape (2, 1), dtype int64'
        ],
        'operation': 'indices = argmax(input, axis=-1, keepdims=True)',
        'category': 'reduction'
    },

    'cumsum': {
        'Input Tensors': [
            'input: shape (2, 2048), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (2, 2048), dtype float32'
        ],
        'operation': 'output = cumsum(input, axis=-1)',
        'category': 'reduction'
    }


}

