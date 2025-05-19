import torch

def standard_matmul_reference(A, B):
    """
    Performs standard matrix multiplication (C = A * B).

    Args:
        A (torch.Tensor): Input matrix A of shape (M, K)
        B (torch.Tensor): Input matrix B of shape (K, N)

    Returns:
        torch.Tensor: Output matrix C of shape (M, N)
    """
    return torch.matmul(A, B)

def get_inputs():
    M = 1024
    K = 4096
    N = 2048
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]
