import torch

def transposed_matmul_reference(A, B):
    """
    Performs matrix multiplication with transposed A (C = A.T * B).

    Args:
        A (torch.Tensor): Input matrix of shape (K, M)
        B (torch.Tensor): Input matrix of shape (K, N)

    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    """
    return torch.matmul(A.T, B)

def get_inputs():
    M = 1024
    K = 4096
    N = 2048
    A = torch.randn(K, M)
    B = torch.randn(K, N)
    return [A, B]
