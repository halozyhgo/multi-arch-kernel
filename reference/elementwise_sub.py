import torch
def elementwise_sub_reference(x, y):
    return torch.sub(x, y)

def get_inputs():
    return [torch.randn(2,2048), torch.randn(2,2048)]