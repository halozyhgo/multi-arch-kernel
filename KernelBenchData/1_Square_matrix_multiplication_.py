import torch

def square_matmul_reference(A, B):
    """
    Performs square matrix multiplication (C = A * B).

    Args:
        A (torch.Tensor): Input matrix A of shape (N, N)
        B (torch.Tensor): Input matrix B of shape (N, N)

    Returns:
        torch.Tensor: Output matrix C of shape (N, N)
    """
    return torch.matmul(A, B)

def get_inputs():
    N = 2048
    A = torch.randn(N, N)
    B = torch.randn(N, N)
    return [A, B]
