import torch

def tensor_4d_matmul_reference(A, B):
    """
    Performs 4D tensor-matrix multiplication: C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]

    Args:
        A (torch.Tensor): Input 4D tensor of shape (b, i, j, l)
        B (torch.Tensor): Input matrix of shape (l, k)

    Returns:
        torch.Tensor: Output 4D tensor of shape (b, i, j, k)
    """
    return torch.einsum("bijl,lk->bijk", A, B)

def get_inputs():
    b = 16
    i = 256
    j = 512
    l = 256
    k = 768
    A = torch.randn(b, i, j, l)
    B = torch.randn(l, k)
    return [A, B]
