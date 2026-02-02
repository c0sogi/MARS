import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RepVGGBlock(nn.Module):
    """
    Re-parameterizable Block (RepVGG Style).
    Consists of parallel 3x3 Conv, 1x1 Conv, and Identity branches during training.
    Fuses into a single 3x3 Conv during inference.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Ensure groups is valid
        if out_channels % groups != 0 or in_channels % groups != 0:
            # Fallback to groups=1 if channels are not divisible
            self.groups = 1

        self.activation = nn.ReLU(inplace=True)
        # SE Block removed (Cite solution_lesson_node_00013)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=self.groups,
                bias=True,
            )
        else:
            # Branch 1: 3x3 Conv + BN
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # Branch 2: 1x1 Conv + BN
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=self.groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # Branch 3: Identity (BN only) if dims match
            if in_channels == out_channels and stride == 1:
                self.branch_identity = nn.BatchNorm2d(out_channels)
            else:
                self.branch_identity = None

    def forward(self, x):
        if self.deploy:
            x = self.fused_conv(x)
            x = self.activation(x)
            return x

        # Multi-branch training forward
        out_3x3 = self.branch_3x3(x)
        out_1x1 = self.branch_1x1(x)

        out = out_3x3 + out_1x1

        if self.branch_identity is not None:
            out += self.branch_identity(x)

        out = self.activation(out)
        return out

    def switch_to_deploy(self):
        """
        Fuses the parallel branches into a single convolutional layer.
        """
        if self.deploy:
            return

        # 1. Get equivalent kernel and bias for 3x3 branch
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.branch_3x3)

        # 2. Get equivalent kernel and bias for 1x1 branch (padded to 3x3)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernel_1x1 = self._pad_1x1_to_3x3_tensor(kernel_1x1)

        # 3. Get equivalent kernel and bias for identity branch
        kernel_id, bias_id = 0, 0
        if self.branch_identity is not None:
            kernel_id, bias_id = self._fuse_bn_identity(self.branch_identity)

        # 4. Sum parameters
        final_kernel = kernel_3x3 + kernel_1x1 + kernel_id
        final_bias = bias_3x3 + bias_1x1 + bias_id

        # 5. Create new fused layer
        self.fused_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.fused_conv.weight.data = final_kernel
        self.fused_conv.bias.data = final_bias

        # 6. Remove old branches
        del self.branch_3x3
        del self.branch_1x1
        if hasattr(self, "branch_identity"):
            del self.branch_identity

        self.deploy = True

    def _fuse_bn_tensor(self, branch):
        """
        Generates fused kernel and bias from a Conv-BN sequence.
        """
        conv = branch[0]
        bn = branch[1]

        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_identity(self, bn):
        """
        Generates fused kernel and bias from a standalone BN (Identity branch).
        Creates an identity convolution kernel scaled by BN parameters.
        """
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # Construct identity kernel for grouped convolution
        # Shape: (out_channels, in_channels/groups, 3, 3)
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            self.out_channels, input_dim, 3, 3, device=bn.weight.device
        )

        for i in range(self.out_channels):
            # The channel index within the group
            input_idx = i % input_dim
            kernel_value[i, input_idx, 1, 1] = 1

        return kernel_value * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        """
        Pads a 1x1 kernel to 3x3.
        """
        if kernel_1x1 is None:
            return 0
        return F.pad(kernel_1x1, [1, 1, 1, 1])


class RepDownsample(nn.Module):
    """
    Re-parameterizable Downsampling Block.
    Parallel Stride-2 3x3 Conv and Stride-2 1x1 Conv.
    """

    def __init__(self, in_channels, out_channels, stride=2, groups=32, deploy=False):
        super(RepDownsample, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        if out_channels % groups != 0 or in_channels % groups != 0:
            self.groups = 1

        self.activation = nn.ReLU(inplace=True)

        if deploy:
            self.fused_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=self.groups,
                bias=True,
            )
        else:
            self.branch_3x3 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.branch_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=self.groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        if self.deploy:
            return self.activation(self.fused_conv(x))

        return self.activation(self.branch_3x3(x) + self.branch_1x1(x))

    def switch_to_deploy(self):
        if self.deploy:
            return

        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.branch_3x3)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.branch_1x1)
        kernel_1x1 = self._pad_1x1_to_3x3_tensor(kernel_1x1)

        final_kernel = kernel_3x3 + kernel_1x1
        final_bias = bias_3x3 + bias_1x1

        self.fused_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.fused_conv.weight.data = final_kernel
        self.fused_conv.bias.data = final_bias

        del self.branch_3x3
        del self.branch_1x1
        self.deploy = True

    def _fuse_bn_tensor(self, branch):
        conv = branch[0]
        bn = branch[1]
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _pad_1x1_to_3x3_tensor(self, kernel_1x1):
        return F.pad(kernel_1x1, [1, 1, 1, 1])
