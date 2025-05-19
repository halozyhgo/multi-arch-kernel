import torch

def matrix_vec_mul_reference(A, B):
    """
    Performs matrix-vector multiplication (C = A * B).

    Args:
        A (torch.Tensor): Input matrix of shape (M, K)
        B (torch.Tensor): Input vector of shape (K, 1)

    Returns:
        torch.Tensor: Output vector of shape (M, 1)
    """
    return torch.matmul(A, B)

def get_inputs():
    M = 256
    K = 131072
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]
