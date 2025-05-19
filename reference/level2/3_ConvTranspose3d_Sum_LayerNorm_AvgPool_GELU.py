import torch
import torch.nn.functional as F

def conv_transpose_sum_norm_pool_gelu_reference(x, weight, sum_weight, 
                                              stride=1, padding=0, output_padding=0, groups=1,
                                              norm_shape=None, pool_kernel_size=2):
    """
    Performs 3D transposed convolution, sum, layer norm, average pooling and GELU.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, D, H, W)
        weight (torch.Tensor): Convolution weights of shape (C_in, C_out, kD, kH, kW)
        sum_weight (float): Weight for the sum operation
        stride, padding, output_padding, groups: ConvTranspose parameters
        norm_shape: Shape for layer normalization
        pool_kernel_size: Size for average pooling

    Returns:
        torch.Tensor: Output tensor after all operations
    """
    x = F.conv_transpose3d(x, weight, None, stride, padding, output_padding, groups)
    x = x + sum_weight
    x = F.layer_norm(x, norm_shape)
    x = F.avg_pool3d(x, kernel_size=pool_kernel_size)
    x = F.gelu(x)
    return x

def get_inputs():
    batch_size = 128
    in_channels = 32
    out_channels = 64
    depth, height, width = 16, 32, 32
    kernel_size = (3, 3, 3)
    stride = (2, 2, 2)
    padding = (1, 1, 1)
    output_padding = (1, 1, 1)
    sum_weight = 1.0
    norm_shape = (out_channels,)
    pool_kernel_size = (2, 2, 2)
    
    x = torch.randn(batch_size, in_channels, depth, height, width)
    weight = torch.randn(in_channels, out_channels, *kernel_size)
    
    return [x, weight, sum_weight, stride, padding, output_padding, groups, norm_shape, pool_kernel_size]
