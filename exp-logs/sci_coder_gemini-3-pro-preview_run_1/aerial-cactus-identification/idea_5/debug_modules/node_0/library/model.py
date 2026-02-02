import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper function to create a Conv2d followed by a BatchNorm2d.
    """
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """
    RepVGG Block that supports structural re-parameterization.

    Training:
        Output = ReLU( (Conv3x3+BN)(x) + (Conv1x1+BN)(x) + BN(x) )
    Inference (after switch_to_deploy):
        Output = ReLU( (FusedConv3x3+Bias)(x) )
    """

    def __init__(self, in_channels, out_channels, stride=1, padding=1, deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.non_linearity = nn.ReLU()

        if deploy:
            # Inference structure: Single 3x3 Conv
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=True,
            )
        else:
            # Training structure: Multi-branch
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.non_linearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.non_linearity(
            self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        )

    def get_equivalent_kernel_bias(self):
        """
        Calculates the fused 3x3 kernel and bias from the 3 branches.
        """
        # 1. Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # 2. Fuse 1x1 branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # Pad 1x1 kernel to 3x3 (center aligned)
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)

        # 3. Fuse Identity branch
        kernelid, biasid = self._get_kernel_bias_identity()

        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        # Pad (left, right, top, bottom) -> (1, 1, 1, 1) to turn 1x1 into 3x3
        return F.pad(kernel1x1, [1, 1, 1, 1])

    def _get_kernel_bias_identity(self):
        if self.rbr_identity is None:
            return 0, 0

        # Identity is equivalent to a convolution with an identity matrix as the kernel
        kernel_value = torch.zeros(
            self.in_channels,
            self.in_channels,
            3,
            3,
            device=self.rbr_identity.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[i, i, 1, 1] = 1

        # Fuse the Identity "Conv" with its BatchNorm
        running_mean = self.rbr_identity.running_mean
        running_var = self.rbr_identity.running_var
        gamma = self.rbr_identity.weight
        beta = self.rbr_identity.bias
        eps = self.rbr_identity.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # Fused Kernel: IdentityKernel * (gamma/std)
        # Fused Bias: beta - mean * (gamma/std)
        return kernel_value * t, beta - running_mean * gamma / std

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        Transforms the block from training mode to inference mode by fusing layers.
        """
        if hasattr(self, "rbr_reparam"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            bias=True,
        )

        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches to save memory
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True


class RepVGGCactus(nn.Module):
    """
    Custom RepVGG-style architecture for Cactus Identification (32x32 images).
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, deploy=False):
        super(RepVGGCactus, self).__init__()

        # Channels: [64, 128, 256, 512]
        channels = Config.MODEL_CHANNELS

        # Stem: 3 -> 64, stride 1. Conservative stem (no pooling, no stride 2)
        self.stage0 = RepVGGBlock(
            in_channels=3, out_channels=channels[0], stride=1, deploy=deploy
        )

        # Stage 1: 64 -> 128 (Downsample)
        self.stage1 = nn.Sequential(
            RepVGGBlock(channels[0], channels[1], stride=2, deploy=deploy),
            RepVGGBlock(channels[1], channels[1], stride=1, deploy=deploy),
        )

        # Stage 2: 128 -> 256 (Downsample)
        self.stage2 = nn.Sequential(
            RepVGGBlock(channels[1], channels[2], stride=2, deploy=deploy),
            RepVGGBlock(channels[2], channels[2], stride=1, deploy=deploy),
        )

        # Stage 3: 256 -> 512 (Downsample)
        self.stage3 = nn.Sequential(
            RepVGGBlock(channels[2], channels[3], stride=2, deploy=deploy),
            RepVGGBlock(channels[3], channels[3], stride=1, deploy=deploy),
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(channels[3], num_classes)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    def reparameterize(self):
        """
        Switch entire network to inference mode by fusing all blocks.
        """
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
