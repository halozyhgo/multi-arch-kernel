import torch

def matmul_with_transposed_both_reference(A, B):
    """
    Performs matrix multiplication with both matrices transposed (C = A.T * B.T).

    Args:
        A (torch.Tensor): Input matrix of shape (K, M)
        B (torch.Tensor): Input matrix of shape (N, K)

    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    """
    return torch.matmul(A.T, B.T)

def get_inputs():
    M = 1024
    K = 4096
    N = 2048
    A = torch.randn(K, M)
    B = torch.randn(N, K)
    return [A, B]
