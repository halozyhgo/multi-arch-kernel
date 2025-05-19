import torch

def conv_transpose_bias_clamp_scale_reference(x, weight, bias, scaling_factor, 
                                            stride=1, padding=0, output_padding=0, groups=1):
    """
    Performs transposed convolution, adds bias, clamps, scales, clamps again, and divides.

    Args:
        x (torch.Tensor): Input tensor of shape (N, C_in, H, W)
        weight (torch.Tensor): Convolution weights of shape (C_in, C_out, kH, kW)
        bias (torch.Tensor): Bias tensor of shape (C_out, 1, 1)
        scaling_factor (float): Scaling factor for the operation
        stride, padding, output_padding, groups: ConvTranspose parameters

    Returns:
        torch.Tensor: Output tensor after all operations
    """
    x = torch.nn.functional.conv_transpose2d(x, weight, None, stride, padding, output_padding, groups)
    x = x + bias
    x = torch.clamp(x, min=0.0, max=1.0)
    x = x * scaling_factor
    x = torch.clamp(x, min=0.0, max=1.0)
    x = x / scaling_factor
    return x

def get_inputs():
    batch_size = 128
    in_channels = 3
    out_channels = 16
    height, width = 32, 32
    kernel_size = 3
    stride = 2
    padding = 1
    output_padding = 1
    scaling_factor = 2.0
    
    x = torch.randn(batch_size, in_channels, height, width)
    weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size)
    bias = torch.randn(out_channels, 1, 1)
    
    return [x, weight, bias, scaling_factor]
