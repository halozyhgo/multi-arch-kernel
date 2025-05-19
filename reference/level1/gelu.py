import torch
def gelu_reference(input):
    return torch.nn.functional.gelu(input)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
