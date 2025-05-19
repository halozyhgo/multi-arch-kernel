import torch

def symmetric_matmul_reference(A, B):
    """
    Performs matrix multiplication of two symmetric matrices (C = A * B).

    Args:
        A (torch.Tensor): Input matrix A, shape (N, N), symmetric
        B (torch.Tensor): Input matrix B, shape (N, N), symmetric

    Returns:
        torch.Tensor: Output matrix C, shape (N, N)
    """
    return torch.matmul(A, B)

def get_inputs():
    N = 4096
    A = torch.randn(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.randn(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]
