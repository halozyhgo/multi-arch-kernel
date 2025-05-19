import torch
def softmax_reference(input, dim=1):
    return torch.softmax(input, dim=dim)

def get_inputs():
    batch_size = 16
    dim = 16384
    x = torch.randn(batch_size, dim)
    return [x]
