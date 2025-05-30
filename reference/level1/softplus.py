import torch
def softplus_reference(input):
    return torch.nn.functional.softplus(input)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
