import torch
import math

def min_gpt_new_gelu_reference(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def get_inputs():
    batch_size = 2000
    dim = 2000
    x = torch.randn(batch_size, dim)
    return [x]
