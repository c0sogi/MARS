import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.utils import set_seed


def transI_fusebn(kernel, bias, bn):
    """
    Fuses a Convolution kernel/bias with a BatchNorm layer.
    """
    gamma = bn.weight
    beta = bn.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps

    std = (running_var + eps).sqrt()
    t = gamma / std

    if bias is None:
        bias = torch.zeros(kernel.shape[0], device=kernel.device)

    # Reshape t to (C_out, 1, 1, 1) for broadcasting against (C_out, C_in/g, K, K)
    fused_kernel = kernel * t.reshape(-1, 1, 1, 1)
    fused_bias = beta + (bias - running_mean) * t

    return fused_kernel, fused_bias


def transII_addbranch(kernels, biases):
    """
    Sums multiple kernels and biases.
    """
    return sum(kernels), sum(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    """
    Merges a sequential 1x1 Conv -> KxK Conv into a single KxK Conv.
    k1: (C_mid, C_in/g, 1, 1)
    k2: (C_out, C_mid/g, K, K)
    """
    # 1. Convolve kernels
    # We treat k1 as the input to the convolution defined by k2.
    # k1 shape is (C_mid, C_in/g, 1, 1).
    # We permute it to (C_in/g, C_mid, 1, 1) to treat C_in/g as the batch dimension
    # and C_mid as the channel dimension for F.conv2d.
    k1_t = k1.permute(1, 0, 2, 3)

    # k2 shape is (C_out, C_mid/g, K, K).
    # F.conv2d with groups=groups will split C_mid into groups.
    k_eq = F.conv2d(k1_t, k2, groups=groups)

    # Output is (C_in/g, C_out, K, K). Permute back to (C_out, C_in/g, K, K).
    k_eq = k_eq.permute(1, 0, 2, 3)

    # 2. Convolve bias
    # The bias b1 from the first conv is effectively added to the input of the second conv.
    # Its contribution to the final bias is sum(k2 * b1).
    # k2_sum shape: (C_out, C_mid/g)
    k2_sum = k2.sum(dim=(2, 3))

    # We use 1x1 convolution to perform the group-wise matrix-vector multiplication
    # b1: (C_mid) -> (1, C_mid, 1, 1)
    # weight: k2_sum -> (C_out, C_mid/g, 1, 1)
    b_hat = F.conv2d(
        b1.reshape(1, -1, 1, 1), k2_sum.unsqueeze(-1).unsqueeze(-1), groups=groups
    )
    b_hat = b_hat.reshape(-1)

    return k_eq, b_hat + b2


def transIV_avg(channels, kernel_size, groups):
    """
    Creates an equivalent convolution kernel for Average Pooling.
    """
    # AvgPool is channel-wise (depthwise), so it acts like groups=channels.
    # However, to add it to a grouped conv with `groups` < `channels`,
    # we construct a kernel of shape (channels, channels/groups, K, K).
    # The kernel is sparse: only the channel corresponding to identity within the group is non-zero.

    weight = torch.zeros(channels, channels // groups, kernel_size, kernel_size)
    val = 1.0 / (kernel_size * kernel_size)

    for c in range(channels):
        # The index of the input channel within the group
        ic = c % (channels // groups)
        weight[c, ic, :, :] = val

    return weight


class DBBGroupedConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(DBBGroupedConv, self).__init__()
        self.deploy = False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias_param = bias

        # 1. Origin Branch (KxK)
        self.dbb_origin = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        self.dbb_bn = nn.BatchNorm2d(out_channels)

        # 2. 1x1 Branch
        # Only valid if K > 1. We assume "same" padding logic relative to KxK center.
        if kernel_size > 1:
            pad_1x1 = padding - kernel_size // 2
            # Ensure valid padding and matching dimensions
            if pad_1x1 >= 0:
                self.dbb_1x1 = nn.Conv2d(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    pad_1x1,
                    groups=groups,
                    bias=False,
                )
                self.dbb_1x1_bn = nn.BatchNorm2d(out_channels)
            else:
                self.dbb_1x1 = None
        else:
            self.dbb_1x1 = None

        # 3. Avg Branch
        # Only valid if dimensions match (Average Pooling preserves channels)
        # and K > 1 (otherwise it's just a scaled identity, covered by 1x1).
        if kernel_size > 1 and in_channels == out_channels:
            self.dbb_avg = nn.AvgPool2d(kernel_size, stride, padding)
            self.dbb_avg_bn = nn.BatchNorm2d(out_channels)
        else:
            self.dbb_avg = None

        # 4. 1x1 - KxK Branch
        # We use in_channels as the internal width.
        if kernel_size > 1:
            mid_channels = in_channels
            self.dbb_1x1_kxk_1 = nn.Conv2d(
                in_channels, mid_channels, 1, 1, 0, groups=groups, bias=False
            )
            self.dbb_1x1_kxk_1_bn = nn.BatchNorm2d(mid_channels)
            self.dbb_1x1_kxk_2 = nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            )
            self.dbb_1x1_kxk_2_bn = nn.BatchNorm2d(out_channels)
        else:
            self.dbb_1x1_kxk_1 = None

    def get_equivalent_kernel_bias(self):
        # Start with Origin
        k_origin, b_origin = transI_fusebn(
            self.dbb_origin.weight, self.dbb_origin.bias, self.dbb_bn
        )
        k_sum = k_origin
        b_sum = b_origin

        # Add 1x1
        if self.dbb_1x1 is not None:
            k_1x1, b_1x1 = transI_fusebn(
                self.dbb_1x1.weight, self.dbb_1x1.bias, self.dbb_1x1_bn
            )
            # Pad 1x1 kernel to KxK
            pad = (self.kernel_size - 1) // 2
            k_1x1 = F.pad(k_1x1, (pad, pad, pad, pad))
            k_sum += k_1x1
            b_sum += b_1x1

        # Add Avg
        if self.dbb_avg is not None:
            k_avg = transIV_avg(self.out_channels, self.kernel_size, self.groups).to(
                k_origin.device
            )
            k_avg, b_avg = transI_fusebn(k_avg, None, self.dbb_avg_bn)
            k_sum += k_avg
            b_sum += b_avg

        # Add 1x1-KxK
        if self.dbb_1x1_kxk_1 is not None:
            k1, b1 = transI_fusebn(
                self.dbb_1x1_kxk_1.weight,
                self.dbb_1x1_kxk_1.bias,
                self.dbb_1x1_kxk_1_bn,
            )
            k2, b2 = transI_fusebn(
                self.dbb_1x1_kxk_2.weight,
                self.dbb_1x1_kxk_2.bias,
                self.dbb_1x1_kxk_2_bn,
            )
            k_seq, b_seq = transIII_1x1_kxk(k1, b1, k2, b2, self.groups)
            k_sum += k_seq
            b_sum += b_seq

        return k_sum, b_sum

    def switch_to_deploy(self):
        if self.deploy:
            return

        k, b = self.get_equivalent_kernel_bias()

        # Create new single layer
        self.dbb_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            bias=True,
        )
        self.dbb_reparam.weight.data = k
        self.dbb_reparam.bias.data = b

        # Remove old branches to save memory
        del self.dbb_origin
        del self.dbb_bn
        if self.dbb_1x1 is not None:
            del self.dbb_1x1
            del self.dbb_1x1_bn
        if self.dbb_avg is not None:
            del self.dbb_avg
            del self.dbb_avg_bn
        if self.dbb_1x1_kxk_1 is not None:
            del self.dbb_1x1_kxk_1
            del self.dbb_1x1_kxk_1_bn
            del self.dbb_1x1_kxk_2
            del self.dbb_1x1_kxk_2_bn

        self.deploy = True

    def forward(self, x):
        if self.deploy:
            return self.dbb_reparam(x)

        # Origin
        out = self.dbb_bn(self.dbb_origin(x))

        # 1x1
        if self.dbb_1x1 is not None:
            out += self.dbb_1x1_bn(self.dbb_1x1(x))

        # Avg
        if self.dbb_avg is not None:
            out += self.dbb_avg_bn(self.dbb_avg(x))

        # 1x1-KxK
        if self.dbb_1x1_kxk_1 is not None:
            x_seq = self.dbb_1x1_kxk_1_bn(self.dbb_1x1_kxk_1(x))
            out += self.dbb_1x1_kxk_2_bn(self.dbb_1x1_kxk_2(x_seq))

        return out
