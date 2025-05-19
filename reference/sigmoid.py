import torch
def sigmoid_reference(x):
    return torch.sigmoid(x)

def get_inputs():
    return [torch.randn(2,2048)]