import torch
def elementwise_add_reference(x, y):
    return torch.add(x, y)

def get_inputs():
    return [torch.randn(2,2048), torch.randn(2,2048)]