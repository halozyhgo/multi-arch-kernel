import torch

def diag_matmul_reference(A, B):
    """
    Performs matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B

    Args:
        A (torch.Tensor): 1D tensor representing the diagonal. Shape: (N,)
        B (torch.Tensor): 2D matrix. Shape: (N, M)

    Returns:
        torch.Tensor: Result of multiplication. Shape: (N, M)
    """
    return torch.diag(A) @ B

def get_inputs():
    M = 4096
    N = 4096
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]
