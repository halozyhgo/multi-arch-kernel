import torch

def upper_triangular_matmul_reference(A, B):
    """
    Performs matrix multiplication for upper triangular matrices (C = A * B).

    Args:
        A (torch.Tensor): Upper triangular matrix, shape (N, N)
        B (torch.Tensor): Upper triangular matrix, shape (N, N)

    Returns:
        torch.Tensor: Upper triangular result matrix, shape (N, N)
    """
    return torch.triu(torch.matmul(A, B))

def get_inputs():
    N = 4096
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]
