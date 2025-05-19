import torch

def irregular_matmul_reference(A, B):
    """
    Performs matrix multiplication with irregular shapes (C = A * B).

    Args:
        A (torch.Tensor): Input matrix of shape (M, K)
        B (torch.Tensor): Input matrix of shape (K, N)

    Returns:
        torch.Tensor: Output matrix of shape (M, N)
    """
    return torch.matmul(A, B)

def get_inputs():
    M = 8205
    K = 2949
    N = 5921
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]
