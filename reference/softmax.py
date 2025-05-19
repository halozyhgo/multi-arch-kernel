import torch
def softmax_reference(x):
    return torch.softmax(x, dim=1)

def get_inputs():
    return [torch.randn(2,2048)]