import os
'''
This document is used to statistically analyze the basic information of operators in level 1, 
input/output information, and other contents.
'''

ref_level1_dataset = {
    'InstanceNorm2d': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype float32',
            'num_features: scalar'
        ],
        'Output Tensors': [
            'output: shape (16, 64, 256, 256), dtype float32'
        ],
        'operation': 'output = InstanceNorm2d(input, num_features)',
        'category': 'normalization'
    },
    'argmax': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32',
            'dim: scalar, value 1'
        ],
        'Output Tensors': [
            'output: shape (16, 256), dtype int64'
        ],
        'operation': 'output = argmax(x, dim)',
        'category': 'math'
    },
    'argmin': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32',
            'dim: scalar, value 1'
        ],
        'Output Tensors': [
            'output: shape (16, 256), dtype int64'
        ],
        'operation': 'output = argmin(x, dim)',
        'category': 'math'
    },
    'avgpool1d': {
        'Input Tensors': [
            'input: shape (16, 32, 128), dtype float32',
            'kernel_size: scalar, value 4',
            'stride: scalar, value 2',
            'padding: scalar, value 1'
        ],
        'Output Tensors': [
            'output: shape (16, 32, ((128 + 2*1 - 4) // 2 + 1)), dtype float32'
        ],
        'operation': 'output = nn.AvgPool1d(kernel_size, stride, padding)(input)',
        'category': 'pooling'
    },
    'avgpool2d': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype float32',
            'kernel_size: scalar, value 3',
            'stride: scalar, value None',
            'padding: scalar, value 0'
        ],
        'Output Tensors': [
            'output: shape ((16, 64, ((256 + 2*0 - 3) // (stride or 3) + 1), ((256 + 2*0 - 3) // (stride or 3) + 1))), dtype float32'
        ],
        'operation': 'output = nn.AvgPool2d(kernel_size, stride, padding)(input)',
        'category': 'pooling'
    },
    'avgpool3d': {
        'Input Tensors': [
            'input: shape (16, 32, 64, 64, 64), dtype float32',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 2',
            'padding: scalar, value 1'
        ],
        'Output Tensors': [
            'output: shape (16, 32, ((64 + 2*1 - 3) // 2 + 1), ((64 + 2*1 - 3) // 2 + 1), ((64 + 2*1 - 3) // 2 + 1)), dtype float32'
        ],
        'operation': 'output = nn.AvgPool3d(kernel_size, stride, padding)(input)',
        'category': 'pooling'
    },
    'batched_matmul': {
        'Input Tensors': [
            'A: shape (128, 128, 256), dtype float32',
            'B: shape (128, 256, 512), dtype float32'
        ],
        'Output Tensors': [
            'C: shape (128, 128, 512), dtype float32'
        ],
        'operation': 'C = torch.bmm(A, B)',
        'category': 'blas'
    },
    'batchnorm': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype float32',
            'num_features: scalar'
        ],
        'Output Tensors': [
            'output: shape (16, 64, 256, 256), dtype float32'
        ],
        'operation': 'output = nn.BatchNorm2d(num_features)(input)',
        'category': 'normalization'
    },
    'conv1d': {
        'Input Tensors': [
            'x: shape (16, 3, 512), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((512 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv1d_dilated_strided': {
        'Input Tensors': [
            'x: shape (16, 3, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 3',
            'dilation: scalar, value 4',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*0 - 4*(3 - 1) - 1) // 3 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv1d(in_channels, out_channels, kernel_size, stride, dilation, bias)(x)',
        'category': 'convolution'
    },
    'conv2d': {
        'Input Tensors': [
            'x: shape (16, 3, 128, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((128 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv2d_asym': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 128), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: tuple, value (3, 5)',
            'stride: tuple, value (1, 1)',
            'padding: tuple, value (0, 0)',
            'dilation: tuple, value (1, 1)',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((128 + 2*0 - 1*(5 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv2d_asym_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: tuple, value (3, 5)',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(5 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv2d_square': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv2d_square_input_asym_kernel_dilated_padded': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: tuple, value (3, 5)',
            'stride: scalar, value 1',
            'padding: tuple, value (1, 2)',
            'dilation: tuple, value (2, 1)',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*1 - 2*(3 - 1) - 1) // 1 + 1), ((256 + 2*2 - 1*(5 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)(x)',
        'category': 'convolution'
    },
    'conv3d': {
        'Input Tensors': [
            'x: shape (16, 3, 64, 64, 64), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((64 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((64 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((64 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv3d_asym_input': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256, 10), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((10 + 2*0 - 1*(1 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv3d_asym_input_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 16, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: tuple, value (3, 5, 7)',
            'stride: tuple, value (1, 1, 1)',
            'padding: tuple, value (0, 0, 0)',
            'dilation: tuple, value (1, 1, 1)',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((16 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(5 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(7 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv3d_asym_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 64, 64, 64), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: tuple, value (3, 5, 7)',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'groups: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((64 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((64 + 2*0 - 1*(5 - 1) - 1) // 1 + 1), ((64 + 2*0 - 1*(7 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'output = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv_depthwise2d_asym_input_asym_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 128, 256), dtype float32',
            'in_channels: scalar, value 3',
            'kernel_size_h: scalar, value 3',
            'kernel_size_w: scalar, value 5',
            'stride_h: scalar, value 1',
            'stride_w: scalar, value 1',
            'padding_h: scalar, value 0',
            'padding_w: scalar, value 0',
            'dilation_h: scalar, value 1',
            'dilation_w: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 3, ((128 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(5 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'conv = nn.Conv2d(in_channels, in_channels, (kernel_size_h, kernel_size_w), stride=(stride_h, stride_w), padding=(padding_h, padding_w), dilation=(dilation_h, dilation_w), groups=in_channels, bias=bias); output = conv(x)',
        'category': 'convolution'
    },
    'conv_depthwise2d_asym_input_square_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 128, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 3',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 3, ((128 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'conv = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride, padding, groups=in_channels, bias=bias); output = conv(x)',
        'category': 'convolution'
    },
    'conv_depthwise2d_square_input_asym_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 3, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(1 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'conv = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias); output = conv(x)',
        'category': 'convolution'
    },
    'conv_depthwise2d_square_input_kernel': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 3, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias); output = conv(x)',
        'category': 'convolution'
    },
    'conv_depthwise_separable_2d': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, value 3',
            'out_channels: scalar, value 64',
            'kernel_size: scalar, value 3',
            'stride: scalar, value 1',
            'padding: scalar, value 0',
            'dilation: scalar, value 1',
            'bias: bool, value False'
        ],
        'Output Tensors': [
            'output: shape (16, 64, ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1), ((256 + 2*0 - 1*(3 - 1) - 1) // 1 + 1)), dtype float32'
        ],
        'operation': 'x = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)(x); x = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)(x); output = x',
        'category': 'convolution'
    },
    'conv_pointwise_2d': {
        'Input Tensors': [
            'x: shape (16, 3, 256, 256), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'bias: scalar, dtype bool'
        ],
        'Output Tensors': [
            'z: shape (16, 64, 256, 256), dtype float32'
        ],
        'operation': 'z = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose1d': {
        'Input Tensors': [
            'x: shape (16, 64, 128), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: scalar, dtype int',
            'stride: scalar, dtype int (default 1)',
            'padding: scalar, dtype int (default 0)',
            'output_padding: scalar, dtype int (default 0)',
            'groups: scalar, dtype int (default 1)',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 3, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose1d_asym_input_square_kernel_padded': {
        'Input Tensors': [
            'x: shape (16, 32, 128), dtype float32 (default from torch.randn)',
            'in_channels: scalar, dtype int (value = 32)',
            'out_channels: scalar, dtype int (value = 64)',
            'kernel_size: scalar, dtype int (value = 3)',
            'stride: scalar, dtype int (value = 2)',
            'padding: scalar, dtype int (value = 1)',
            'dilation: scalar, dtype int (value = 2)',
            'bias: boolean (value = False)'
        ],
        'Output Tensors': [
            'output: shape calculated by ConvTranspose1d rules, dtype float32'
        ],
        'operation': 'output = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)(x)',
        'category': 'math'
    },

    'conv_transpose1d_dilated': {
        'Input Tensors': [
            'x: shape (16, 32, 128), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: scalar, dtype int',
            'stride: scalar, dtype int (default 1)',
            'padding: scalar, dtype int (default 0)',
            'dilation: scalar, dtype int (default 1)',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d': {
        'Input Tensors': [
            'x: shape (16, 32, 128, 128), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: scalar, dtype int',
            'stride: scalar, dtype int (default 1)',
            'padding: scalar, dtype int (default 0)',
            'output_padding: scalar, dtype int (default 0)',
            'groups: scalar, dtype int (default 1)',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d_asym_input_kernel': {
        'Input Tensors': [
            'x: shape (16, 32, 16, 32), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: tuple (3, 5), dtype int',
            'stride: tuple (1, 1), dtype int (default)',
            'padding: tuple (0, 0), dtype int (default)',
            'output_padding: tuple (0, 0), dtype int (default)',
            'dilation: tuple (1, 1), dtype int (default)',
            'groups: scalar, dtype int (default 1)',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d_asym_input_kernel_padded': {
        'Input Tensors': [
            'x: shape (16, 32, 128, 256), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: tuple (3, 5), dtype int',
            'stride: tuple (1, 1), dtype int (default)',
            'padding: tuple (1, 2), dtype int',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d_asym_input_kernel_strided_grouped_padded_dilated': {
        'Input Tensors': [
            'x: shape (16, 32, 128, 256), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: tuple (3, 5), dtype int',
            'stride: tuple (2, 3), dtype int',
            'padding: tuple (1, 2), dtype int',
            'dilation: tuple (2, 1), dtype int',
            'groups: scalar, dtype int',
            'bias: scalar, dtype bool'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d_asym_input_square_kernel': {
        'Input Tensors': [
            'x: shape (16, 32, 128, 256), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: scalar, dtype int',
            'stride: scalar, dtype int (default 1)',
            'padding: scalar, dtype int (default 0)',
            'output_padding: scalar, dtype int (default 0)',
            'groups: scalar, dtype int (default 1)',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d_asym_input_square_kernel_dilated_padded_strided': {
        'Input Tensors': [
            'x: shape (16, 32, 64, 128), dtype float32',
            'in_channels: scalar, dtype int',
            'out_channels: scalar, dtype int',
            'kernel_size: scalar, dtype int',
            'stride: scalar, dtype int (default 1)',
            'padding: scalar, dtype int (default 0)',
            'dilation: scalar, dtype int (default 1)',
            'bias: scalar, dtype bool (default False)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose2d_asym_kernel': {
        'Input Tensors': [
            'x: shape (16, 32, 128, 128), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 64',
            'kernel_size: tuple (3, 5), dtype int',
            'stride: scalar, dtype int, value 1',
            'padding: scalar, dtype int, value 0',
            'output_padding: scalar, dtype int, value 0',
            'groups: scalar, dtype int, value 1',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *), dtype same as x',
        ],
        'operation': 'z = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_asym': {
        'Input Tensors': [
            'x: shape (16, 32, 16, 32, 64), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 16',
            'kernel_size: tuple (3, 5, 7), dtype int',
            'stride: tuple (1, 1, 1), dtype int',
            'padding: tuple (0, 0, 0), dtype int',
            'output_padding: tuple (0, 0, 0), dtype int',
            'groups: scalar, dtype int, value 1',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 16, *, *, *), dtype same as x'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_asym_input_kernel_strided_grouped': {
        'Input Tensors': [
            'x: shape (16, 32, 16, 32, 64), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 64',
            'kernel_size: tuple (3, 5, 7), dtype int',
            'stride: tuple (2, 2, 2), dtype int',
            'padding: tuple (1, 2, 3), dtype int',
            'output_padding: tuple (1, 1, 1), dtype int',
            'groups: scalar, dtype int, value 4',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *, *), dtype same as x'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_asym_input_square_kernel': {
        'Input Tensors': [
            'x: shape (16, 32, 16, 32, 64), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 16',
            'kernel_size: scalar, dtype int, value 3',
            'stride: scalar, dtype int, value 1',
            'padding: scalar, dtype int, value 0',
            'output_padding: scalar, dtype int, value 0',
            'dilation: scalar, dtype int, value 1',
            'groups: scalar, dtype int, value 1',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 16, *, *, *), dtype same as x'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, dilation=dilation, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_asym_input_square_kernel_strided_grouped': {
        'Input Tensors': [
            'x: shape (16, 32, 16, 32, 32), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 64',
            'kernel_size: scalar, dtype int, value 3',
            'stride: scalar, dtype int, value 2',
            'padding: scalar, dtype int, value 3',
            'output_padding: scalar, dtype int, value 0',
            'groups: scalar, dtype int, value 4',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *, *), dtype same as x'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_asym_kernel': {
        'Input Tensors': [
            'x: shape (16, 32, 64, 64, 64), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 64',
            'kernel_size: tuple (3, 5, 5), dtype int',
            'stride: tuple (1, 1, 1), dtype int',
            'padding: tuple (0, 0, 0), dtype int',
            'output_padding: tuple (0, 0, 0), dtype int',
            'groups: scalar, dtype int, value 1',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *, *), dtype same as x'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_square': {
        'Input Tensors': [
            'x: shape (16, 3, 32, 32, 32), dtype inferred from torch.randn (usually float32)',
            'in_channels: scalar, dtype int, value 3',
            'out_channels: scalar, dtype int, value 64',
            'kernel_size: scalar, dtype int, value 3',
            'stride: scalar, dtype int, value 1',
            'padding: scalar, dtype int, value 0',
            'output_padding: scalar, dtype int, value 0',
            'groups: scalar, dtype int, value 1',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *, *), dtype same as x'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, groups=groups, bias=bias)(x)',
        'category': 'convolution'
    },
    'conv_transpose3d_square_input_kernel_padded_dilated_strided': {
        'Input Tensors': [
            'x: shape (16, 32, 16, 32, 32), dtype float32',
            'in_channels: scalar, dtype int, value 32',
            'out_channels: scalar, dtype int, value 64',
            'kernel_size: scalar, dtype int, value 3',
            'stride: scalar, dtype int, value 2',
            'padding: scalar, dtype int, value 1',
            'dilation: scalar, dtype int, value 2',
            'bias: scalar, dtype bool, value False'
        ],
        'Output Tensors': [
            'z: shape (16, 64, *, *, *), dtype float32'
        ],
        'operation': 'z = nn.ConvTranspose3d(in_channels, out_channels, (kernel_size, kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, bias=bias)(x)',
        'category': 'convolution'
    },
    'cosine_similarity_loss': {
        'Input Tensors': [
            'predictions: shape (128, 4096), dtype inferred from torch.randn (usually float32)',
            'targets: shape (128, 4096), dtype inferred from torch.randn (usually float32)'
        ],
        'Output Tensors': [
            'z: shape (), dtype same as predictions and targets'
        ],
        'operation': 'z = torch.mean(1 - F.cosine_similarity(predictions, targets, dim=1))',
        'category': 'loss'
    },
    'cross_entropy': {
        'Input Tensors': [
            'predictions: shape (4096, 10), dtype inferred from torch.randn (usually float32)',
            'targets: shape (4096,), dtype inferred from torch.randint (usually int64)'
        ],
        'Output Tensors': [
            'z: shape (), dtype same as predictions'
        ],
        'operation': 'z = F.cross_entropy(predictions, targets)',
        'category': 'loss'
    },
    'cumprod': {
        'Input Tensors': [
            'x: shape (128, 4000), dtype inferred from torch.randn (usually float32)',
            'dim: scalar, dtype int, value 1'
        ],
        'Output Tensors': [
            'z: shape (128, 4000), dtype same as x'
        ],
        'operation': 'z = torch.cumprod(x, dim=dim)',
        'category': 'math'
    },
    'cumsum': {
        'Input Tensors': [
            'x: shape (128, 4000), dtype inferred from torch.randn (usually float32)',
            'dim: scalar, dtype int, value 1'
        ],
        'Output Tensors': [
            'z: shape (128, 4000), dtype same as x'
        ],
        'operation': 'z = torch.cumsum(x, dim=dim)',
        'category': 'math'
    },
    'cumsum_exclusive': {
        'Input Tensors': [
            'x: shape (128, 4000), dtype inferred from torch.randn (usually float32)',
            'dim: scalar, dtype int, value 1'
        ],
        'Output Tensors': [
            'z: shape (128, 4000), dtype same as x'
        ],
        'operation': 'exclusive_cumsum = torch.cat((torch.zeros_like(x.select(dim, 0).unsqueeze(dim)), x), dim=dim)[:-1]; z = torch.cumsum(exclusive_cumsum, dim=dim)',
        'category': 'math'
    },
    'cumsum_reverse': {
        'Input Tensors': [
            'x: shape (128, 4000), dtype inferred from torch.randn (usually float32)',
            'dim: scalar, dtype int, value 1'
        ],
        'Output Tensors': [
            'z: shape (128, 4000), dtype same as x'
        ],
        'operation': 'z = torch.cumsum(x.flip(dim), dim=dim).flip(dim)',
        'category': 'math'
    },
    'diag_matmul': {
        'Input Tensors': [
            'A: shape (4096,), dtype inferred from torch.randn (usually float32)',
            'B: shape (4096, 4096), dtype inferred from torch.randn (usually float32)'
        ],
        'Output Tensors': [
            'C: shape (4096, 4096), dtype same as A and B'
        ],
        'operation': 'C = torch.diag(A) @ B',
        'category': 'math'
    },
    'elu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype inferred from torch.randn (usually float32)',
            'alpha: scalar, dtype float, value 1.0'
        ],
        'Output Tensors': [
            'z: shape (16, 16384), dtype same as input'
        ],
        'operation': 'z = torch.nn.functional.elu(input, alpha=alpha)',
        'category': 'activation'
    },
    'frobeniusnorm': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype inferred from torch.randn (usually float32)'
        ],
        'Output Tensors': [
            'z: shape (16, 64, 256, 256), dtype same as input'
        ],
        'operation': 'norm = torch.norm(input, p=\'fro\'); z = input / norm',
        'category': 'math'
    },
    'gelu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype inferred from torch.randn (usually float32)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype same as input'
        ],
        'operation': 'output = torch.nn.functional.gelu(input)',
        'category': 'activation'
    },
    'groupnorm': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype inferred from torch.randn (usually float32)',
            'num_features: scalar, dtype int, value 64',
            'num_groups: scalar, dtype int, value 8'
        ],
        'Output Tensors': [
            'output: shape (16, 64, 256, 256), dtype same as input'
        ],
        'operation': 'gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_features); output = gn(input)',
        'category': 'normalization'
    },
    'hardsigmoid': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype inferred from torch.randn (usually float32)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype same as input'
        ],
        'operation': 'output = torch.nn.functional.hardsigmoid(input)',
        'category': 'activation'
    },
    'hardtanh': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype inferred from torch.randn (usually float32)',
            'min_val: scalar, dtype float, value -1.0',
            'max_val: scalar, dtype float, value 1.0'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype same as input'
        ],
        'operation': 'output = torch.nn.functional.hardtanh(input, min_val=min_val, max_val=max_val)',
        'category': 'activation'
    },
    'hinge_loss': {
        'Input Tensors': [
            'predictions: shape (128, 1), dtype float32',
            'targets: shape (128, 1), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (1,), dtype float32'
        ],
        'operation': 'output = torch.mean(torch.clamp(1 - predictions * targets, min=0))',
        'category': 'loss'
    },
    'huber_loss': {
        'Input Tensors': [
            'predictions: shape (128, 4096), dtype float32',
            'targets: shape (128, 4096), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (), dtype float32'
        ],
        'operation': 'output = F.smooth_l1_loss(predictions, targets)',
        'category': 'loss'
    },
    'irregular_matmul': {
        'Input Tensors': [
            'A: shape (8205, 2949), dtype float32',
            'B: shape (2949, 5921), dtype float32'
        ],
        'Output Tensors': [
            'C: shape (8205, 5921), dtype float32'
        ],
        'operation': 'C = torch.matmul(A, B)',
        'category': 'math'
    },
    'kldiv': {
        'Input Tensors': [
            'predictions: shape (128, 4096), dtype float32',
            'targets: shape (128, 4096), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (), dtype float32'
        ],
        'operation': 'output = F.kl_div(torch.log(predictions), targets, reduction=\'batchmean\')',
        'category': 'loss'
    },
    'l1norm': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = input / torch.sum(torch.abs(input), dim=1, keepdim=True)',
        'category': 'math'
    },
    'l2_normalization': {
        'Input Tensors': [
            'x: shape (16, 16384), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = x / torch.norm(x, p=2, dim=1, keepdim=True)',
        'category': 'math'
    },
    'large_k_matmul': {
        'Input Tensors': [
            'A: shape (256, 131072), dtype float32',
            'B: shape (131072, 256), dtype float32'
        ],
        'Output Tensors': [
            'C: shape (256, 256), dtype float32'
        ],
        'operation': 'C = torch.matmul(A, B)',
        'category': 'math'
    },
    'layernorm': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype float32',
            'normalized_shape: tuple (64, 256, 256)'
        ],
        'Output Tensors': [
            'output: shape (16, 64, 256, 256), dtype float32'
        ],
        'operation': 'output = nn.LayerNorm(normalized_shape=normalized_shape)(input)',
        'category': 'normalization'
    },
    'leaky_relu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.nn.functional.leaky_relu(input, negative_slope=0.01)',
        'category': 'activation'
    },
    'log_softmax': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.log_softmax(input, dim=1)',
        'category': 'activation'
    },
    'lower_triangular_matmul': {
        'Input Tensors': [
            'A: shape (4096, 4096), dtype float32 (assumed default from torch.randn)',
            'B: shape (4096, 4096), dtype float32 (assumed default from torch.randn)'
        ],
        'Output Tensors': [
            'C: shape (4096, 4096), dtype float32 (same as input after matmul and tril)'
        ],
        'operation': 'C = torch.tril(torch.matmul(A, B))',
        'category': 'math'
    },
    'masked_cumsum': {
        'Input Tensors': [
            'x: shape (128, 4000), dtype float32 (default from torch.randn)',
            'mask: shape (128, 4000), dtype bool',
            'dim: scalar, dtype int (value = 1)'
        ],
        'Output Tensors': [
            'output: shape (128, 4000), dtype float32'
        ],
        'operation': 'output = torch.cumsum(x * mask, dim=dim)',
        'category': 'math'
    },
    'matmul_with_transposed_A': {
        'Input Tensors': [
            'A: shape (4096, 1024), dtype float32',
            'B: shape (4096, 2048), dtype float32'
        ],
        'Output Tensors': [
            'C: shape (1024, 2048), dtype float32'
        ],
        'operation': 'C = torch.matmul(A.T, B)',
        'category': 'math'
    },
    'matmul_with_transposed_B': {
        'Input Tensors': [
            'A: shape (1024, 4096), dtype float32',
            'B: shape (2048, 4096), dtype float32'
        ],
        'Output Tensors': [
            'output: shape (1024, 2048), dtype float32'
        ],
        'operation': 'output = torch.matmul(A, B.T)',
        'category': 'math'
    },
    'matmul_with_transposed_both': {
        'Input Tensors': [
            'A: shape (4096, 1024), dtype float32 (default from torch.randn)',
            'B: shape (2048, 4096), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'C: shape (1024, 2048), dtype float32'
        ],
        'operation': 'C = torch.matmul(A.T, B.T)',
        'category': 'math'
    },
    'matrix_scalar_mul': {
        'Input Tensors': [
            'A: shape (16384, 4096), dtype float32 (default from torch.randn)',
            's: scalar, dtype float (value = 3.14)'
        ],
        'Output Tensors': [
            'C: shape (16384, 4096), dtype float32'
        ],
        'operation': 'C = A * s',
        'category': 'math'
    },
    'matrix_vec_mul': {
        'Input Tensors': [
            'A: shape (256, 131072), dtype float32 (default from torch.randn)',
            'B: shape (131072, 1), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'C: shape (256, 1), dtype float32'
        ],
        'operation': 'C = torch.matmul(A, B)',
        'category': 'math'
    },
    'max_reduction': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32 (default from torch.randn)',
            'dim: scalar, dtype int (value = 1)'
        ],
        'Output Tensors': [
            'output: shape (16, 256), dtype float32'
        ],
        'operation': 'output = torch.max(x, dim=dim)[0]',
        'category': 'math'
    },
    'maxpool1d': {
        'Input Tensors': [
            'input: shape (16, 64, 128), dtype float32 (default from torch.randn)',
            'kernel_size: scalar, dtype int (value = 4)',
            'stride: scalar, dtype int (value = 2)',
            'padding: scalar, dtype int (value = 2)',
            'dilation: scalar, dtype int (value = 3)',
            'return_indices: scalar, dtype bool (value = False)'
        ],
        'Output Tensors': [
            'output: shape depends on input and pooling parameters, dtype float32'
        ],
        'operation': 'output = nn.MaxPool1d(kernel_size, stride, padding, dilation, return_indices)(input)',
        'category': 'math'
    },
    'maxpool2d': {
        'Input Tensors': [
            'input: shape (16, 32, 128, 128), dtype float32 (default from torch.randn)',
            'kernel_size: scalar, dtype int (value = 2)',
            'stride: scalar, dtype int (value = 2)',
            'padding: scalar, dtype int (value = 1)',
            'dilation: scalar, dtype int (value = 3)'
        ],
        'Output Tensors': [
            'output: shape depends on input and pooling parameters, dtype float32'
        ],
        'operation': 'output = nn.MaxPool2d(kernel_size, stride, padding, dilation)(input)',
        'category': 'math'
    },
    'maxpool3d': {
        'Input Tensors': [
            'input: shape (16, 32, 64, 64, 64), dtype float32 (default from torch.randn)',
            'kernel_size: scalar, dtype int (value = 3)',
            'stride: scalar, dtype int (value = 2)',
            'padding: scalar, dtype int (value = 1)',
            'dilation: scalar, dtype int (value = 3)',
            'return_indices: scalar, dtype bool (value = False)',
            'ceil_mode: scalar, dtype bool (value = False)'
        ],
        'Output Tensors': [
            'output: shape depends on input and pooling parameters, dtype float32'
        ],
        'operation': 'output = nn.MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(input)',
        'category': 'math'
    },
    'mean_reduction': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32 (default from torch.randn)',
            'dim: scalar, dtype int (value = 1)'
        ],
        'Output Tensors': [
            'output: shape (16, 256), dtype float32'
        ],
        'operation': 'output = torch.mean(x, dim=dim)',
        'category': 'math'
    },
    'min_gpt_new_gelu': {
        'Input Tensors': [
            'x: shape (2000, 2000), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (2000, 2000), dtype float32'
        ],
        'operation': 'output = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))',
        'category': 'math'
    },
    'min_reduction': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32 (default from torch.randn)',
            'dim: scalar, dtype int (value = 1)'
        ],
        'Output Tensors': [
            'output: shape (16, 256), dtype float32'
        ],
        'operation': 'output = torch.min(x, dim=dim)[0]',
        'category': 'math'
    },
    'mse_loss': {
        'Input Tensors': [
            'predictions: shape (128, 4096), dtype float32 (default from torch.randn)',
            'targets: shape (128, 4096), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: scalar, dtype float32'
        ],
        'operation': 'output = torch.mean((predictions - targets) ** 2)',
        'category': 'math'
    },
    'product_reduction': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32 (default from torch.randn)',
            'dim: scalar, dtype int (value = 1)'
        ],
        'Output Tensors': [
            'output: shape (16, 256), dtype float32'
        ],
        'operation': 'output = torch.prod(x, dim=dim)',
        'category': 'math'
    },
    'relu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.relu(input)',
        'category': 'math'
    },
    'rms_norm': {
        'Input Tensors': [
            'input: shape (16, 64, 256, 256), dtype float32 (default from torch.randn)',
            'num_features: scalar, dtype int (value = 64)',
            'eps: scalar, dtype float (value = 1e-5)'
        ],
        'Output Tensors': [
            'output: shape (16, 64, 256, 256), dtype float32'
        ],
        'operation': 'rms = torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + eps); output = input / rms',
        'category': 'math'
    },
    'selu': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.selu(input)',
        'category': 'math'
    },
    'sigmoid': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.sigmoid(input)',
        'category': 'math'
    },
    'small_k_matmul': {
        'Input Tensors': [
            'A: shape (16384, 32), dtype float32 (default from torch.randn)',
            'B: shape (32, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16384, 16384), dtype float32'
        ],
        'operation': 'output = torch.matmul(A, B)',
        'category': 'math'
    },
    'softmax': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)',
            'dim: scalar, dtype int (value = 1)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.softmax(input, dim=dim)',
        'category': 'math'
    },
    'softplus': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.nn.functional.softplus(input)',
        'category': 'math'
    },
    'softsign': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = input / (1 + torch.abs(input))',
        'category': 'math'
    },
    'square_matmul': {
        'Input Tensors': [
            'A: shape (2048, 2048), dtype float32 (default from torch.randn)',
            'B: shape (2048, 2048), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (2048, 2048), dtype float32'
        ],
        'operation': 'output = torch.matmul(A, B)',
        'category': 'math'
    },
    'standard_matmul': {
        'Input Tensors': [
            'A: shape (1024, 4096), dtype float32 (default from torch.randn)',
            'B: shape (4096, 2048), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'C: shape (1024, 2048), dtype float32'
        ],
        'operation': 'C = torch.matmul(A, B)',
        'category': 'math'
    },
    'sum_reduction': {
        'Input Tensors': [
            'x: shape (16, 256, 256), dtype float32 (default from torch.randn)',
            'dim: scalar, dtype int (value = 1)',
            'keepdim: scalar, dtype bool (value = True)'
        ],
        'Output Tensors': [
            'output: shape (16, 1, 256), dtype float32'
        ],
        'operation': 'output = torch.sum(x, dim=dim, keepdim=keepdim)',
        'category': 'math'
    },
    'swish': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = input * torch.sigmoid(input)',
        'category': 'math'
    },
    'symmetric_matmul': {
        'Input Tensors': [
            'A: shape (4096, 4096), dtype float32 (default from torch.randn), symmetric',
            'B: shape (4096, 4096), dtype float32 (default from torch.randn), symmetric'
        ],
        'Output Tensors': [
            'C: shape (4096, 4096), dtype float32'
        ],
        'operation': 'C = torch.matmul(A, B)',
        'category': 'math'
    },
    'tall_skinny_matmul': {
        'Input Tensors': [
            'A: shape (16384, 16), dtype float32 (default from torch.randn)',
            'B: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'C: shape (16384, 16384), dtype float32'
        ],
        'operation': 'C = torch.matmul(A, B)',
        'category': 'math'
    },
    'tanh': {
        'Input Tensors': [
            'input: shape (16, 16384), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 16384), dtype float32'
        ],
        'operation': 'output = torch.tanh(input)',
        'category': 'math'
    },
    'tensor_3d_matmul': {
        'Input Tensors': [
            'A: shape (16, 1024, 2048), dtype float32 (default from torch.randn)',
            'B: shape (2048, 768), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 1024, 768), dtype float32'
        ],
        'operation': 'output = torch.matmul(A, B)',
        'category': 'math'
    },
    'tensor_4d_matmul': {
        'Input Tensors': [
            'A: shape (16, 256, 512, 256), dtype float32 (default from torch.randn)',
            'B: shape (256, 768), dtype float32 (default from torch.randn)'
        ],
        'Output Tensors': [
            'output: shape (16, 256, 512, 768), dtype float32'
        ],
        'operation': 'output = torch.einsum("bijl,lk->bijk", A, B)',
        'category': 'math'
    },
    'triplet_margin': {
        'Input Tensors': [
            'anchor: shape (128, 4096), dtype float32 (default from torch.randn)',
            'positive: shape (128, 4096), dtype float32 (default from torch.randn)',
            'negative: shape (128, 4096), dtype float32 (default from torch.randn)',
            'margin: scalar, dtype float (value = 1.0)'
        ],
        'Output Tensors': [
            'output: scalar, dtype float32'
        ],
        'operation': 'output = nn.TripletMarginLoss(margin=margin)(anchor, positive, negative)',
        'category': 'math'
    },
    'upper_triangular_matmul': {
        'Input Tensors': [
            'A: shape (4096, 4096), dtype float32 (default from torch.randn), upper triangular',
            'B: shape (4096, 4096), dtype float32 (default from torch.randn), upper triangular'
        ],
        'Output Tensors': [
            'C: shape (4096, 4096), dtype float32, upper triangular'
        ],
        'operation': 'C = torch.triu(torch.matmul(A, B))',
        'category': 'math'
    }

}
if __name__ == '__main__':
    namelist = os.listdir('./reference/level1')
    namelist = [name[:-3] for name in namelist if name[-3:] == '.py']
    print(namelist)
    print(len(namelist))
    print()

    dict = ref_level1_dataset
    print(len(dict))
    print(dict.keys())

    print(namelist - dict.keys())
