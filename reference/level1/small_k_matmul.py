import torch

def small_k_matmul_reference(A, B):
    """
    Performs matrix multiplication with small K dimension (C = A * B).

    Args:
        A (torch.Tensor): Input matrix of shape (M, K)
        B (torch.Tensor): Input matrix of shape (K, N)

    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    """
    return torch.matmul(A, B)

def get_inputs():
    M = 16384
    N = 16384
    K = 32
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]
