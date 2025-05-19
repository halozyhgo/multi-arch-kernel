import torch

def batched_matmul_reference(A, B):
    """
    Performs batched matrix multiplication (C = A * B).

    Args:
        A (torch.Tensor): Input tensor of shape (batch_size, m, k)
        B (torch.Tensor): Input tensor of shape (batch_size, k, n)

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, m, n)
    """
    return torch.bmm(A, B)

def get_inputs():
    batch_size = 128
    m = 128
    k = 256
    n = 512
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]
