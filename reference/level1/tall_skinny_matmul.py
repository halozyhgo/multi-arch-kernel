import torch

def tall_skinny_matmul_reference(A, B):
    """
    Performs matrix multiplication with tall/skinny matrices (C = A * B).

    Args:
        A (torch.Tensor): Input matrix of shape (M, N) where M >> N
        B (torch.Tensor): Input matrix of shape (N, M) where M >> N

    Returns:
        torch.Tensor: Output matrix of shape (M, M)
    """
    return torch.matmul(A, B)

def get_inputs():
    M = 16384
    N = 16
    A = torch.randn(M, N)
    B = torch.randn(N, M)
    return [A, B]
