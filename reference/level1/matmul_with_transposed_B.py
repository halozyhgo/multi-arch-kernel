import torch

def matmul_with_transposed_B_reference(A, B):
    """
    Performs matrix multiplication with transposed B (C = A * B.T).

    Args:
        A (torch.Tensor): Input matrix of shape (M, K)
        B (torch.Tensor): Input matrix of shape (N, K)

    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    """
    return torch.matmul(A, B.T)

def get_inputs():
    M = 1024
    K = 4096
    N = 2048
    A = torch.randn(M, K)
    B = torch.randn(N, K)
    return [A, B]
