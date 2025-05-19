import torch

def lower_triangular_matmul_reference(A, B):
    """
    Performs matrix multiplication for lower triangular matrices (C = A * B).

    Args:
        A (torch.Tensor): Lower triangular matrix, shape (N, N)
        B (torch.Tensor): Lower triangular matrix, shape (N, N)

    Returns:
        torch.Tensor: Lower triangular result matrix, shape (N, N)
    """
    return torch.tril(torch.matmul(A, B))

def get_inputs():
    N = 4096
    A = torch.tril(torch.randn(N, N))
    B = torch.tril(torch.randn(N, N))
    return [A, B]
