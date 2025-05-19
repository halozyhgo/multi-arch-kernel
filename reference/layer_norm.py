import torch
from torch.nn import functional as F
def layer_norm_reference(input, gamma, beta):
    return F.layer_norm(
            input, (input.shape[-1],), gamma, beta
        )

def get_inputs():
    return [torch.randn(16,64,256), torch.randn(256,), torch.randn(256,)]
