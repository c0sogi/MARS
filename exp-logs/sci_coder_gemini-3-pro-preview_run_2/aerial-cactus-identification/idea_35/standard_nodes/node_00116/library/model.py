import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper to create a Conv2d followed by BatchNorm2d.
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


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    """

    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        # Ensure reduction doesn't make channels too small
        reduced_channels = max(input_channels // reduction_ratio, 8)
        self.fc1 = nn.Conv2d(input_channels, reduced_channels, 1, bias=True)
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, 1, bias=True)

    def forward(self, x):
        # Squeeze: Global Average Pooling
        scale = F.adaptive_avg_pool2d(x, (1, 1))
        # Excitation
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = torch.sigmoid(scale)
        # Scale
        return x * scale


class RepNeXtBlock(nn.Module):
    """
    RepNeXt Block: A re-parameterizable block combining ResNeXt cardinality with RepVGG efficiency.

    Training: Multi-branch (Grouped 3x3 + Grouped 1x1 + Identity).
    Inference: Fused Grouped 3x3.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=32, deploy=False):
        super(RepNeXtBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Constraint: groups must divide in_channels and out_channels
        # Special case: If input is 3 channels (RGB), we cannot use groups=32. Force groups=1.
        if in_channels == 3:
            self.groups = 1

        # Ensure groups validity (fallback to 1 if invalid)
        if in_channels % self.groups != 0 or out_channels % self.groups != 0:
            self.groups = 1

        padding = 1

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=self.groups,
                bias=True,
            )
        else:
            # Branch 1: Grouped 3x3 Conv + BN
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=self.groups,
            )

            # Branch 2: Grouped 1x1 Conv + BN
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=self.groups,
            )

            # Branch 3: Identity + BN (Only if dimensions match and stride is 1)
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = None

        # SE Block is applied after the addition/fusion
        self.se = SEBlock(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            x = self.rbr_reparam(inputs)
            x = self.se(x)
            return self.activation(x)

        # Calculate Identity branch
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        # Sum branches
        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        x = self.se(x)
        return self.activation(x)

    def get_equivalent_kernel_bias(self):
        """
        Calculates the fused kernel and bias for inference.
        """
        # Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # Fuse 1x1 branch and pad to 3x3
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel1x1 = self._pad_1x1_to_3x3_tensor(kernel1x1)

        # Fuse Identity branch
        kernelid, biasid = self._fuse_identity()

        return kernel3x3 + kernel1x1 + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is 0:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])

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

    def _fuse_identity(self):
        if self.rbr_identity is None:
            return 0, 0

        running_mean = self.rbr_identity.running_mean
        running_var = self.rbr_identity.running_var
        gamma = self.rbr_identity.weight
        beta = self.rbr_identity.bias
        eps = self.rbr_identity.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # Create identity kernel of shape (Out, In/Groups, 3, 3)
        # Since In=Out, input_dim = In/Groups
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(self.in_channels, input_dim, 3, 3)

        # Set center to 1 for corresponding channels
        for i in range(self.in_channels):
            kernel_value[i, i % input_dim, 1, 1] = 1

        # Move to correct device
        if hasattr(self.rbr_dense.conv, "weight"):
            kernel_value = kernel_value.to(self.rbr_dense.conv.weight.device)

        return kernel_value * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        Converts the multi-branch structure to a single convolution.
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
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches to save memory
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        self.deploy = True


class CactusModel(nn.Module):
    """
    Main Architecture: Custom Narrow SE-RepVGG with Multi-Scale Aggregation.
    """

    def __init__(self):
        super(CactusModel, self).__init__()

        channels = Config.MODEL_CHANNELS  # [96, 192, 384]
        groups = Config.GROUPS

        # --- Stage 1 (32x32) ---
        # Block 0: Stem-like projection (3 -> 96). Stride 1 to preserve 32x32.
        self.stage1_0 = RepNeXtBlock(3, channels[0], stride=1, groups=groups)
        # Block 1: Processing (96 -> 96)
        self.stage1_1 = RepNeXtBlock(channels[0], channels[0], stride=1, groups=groups)

        # --- Stage 2 (16x16) ---
        # Block 0: Downsample (96 -> 192). Stride 2.
        self.stage2_0 = RepNeXtBlock(channels[0], channels[1], stride=2, groups=groups)
        # Block 1: Processing (192 -> 192)
        self.stage2_1 = RepNeXtBlock(channels[1], channels[1], stride=1, groups=groups)

        # --- Stage 3 (8x8) ---
        # Block 0: Downsample (192 -> 384). Stride 2.
        self.stage3_0 = RepNeXtBlock(channels[1], channels[2], stride=2, groups=groups)
        # Block 1: Processing (384 -> 384)
        self.stage3_1 = RepNeXtBlock(channels[2], channels[2], stride=1, groups=groups)

        # --- Head (Multi-Scale Aggregation) ---
        # Concatenating GAP from Stage 2 (192) and Stage 3 (384)
        self.fc = nn.Linear(channels[1] + channels[2], Config.NUM_CLASSES)

    def forward(self, x):
        # Stage 1
        x = self.stage1_0(x)
        x = self.stage1_1(x)

        # Stage 2
        x = self.stage2_0(x)
        x = self.stage2_1(x)
        feat2 = x  # Save 16x16 features

        # Stage 3
        x = self.stage3_0(x)
        x = self.stage3_1(x)
        feat3 = x  # Save 8x8 features

        # Multi-Scale Aggregation
        # Global Average Pooling
        gap2 = F.adaptive_avg_pool2d(feat2, (1, 1)).flatten(1)
        gap3 = F.adaptive_avg_pool2d(feat3, (1, 1)).flatten(1)

        # Concatenate
        combined = torch.cat([gap2, gap3], dim=1)

        # Classification
        out = self.fc(combined)
        return out

    def switch_to_deploy(self):
        """
        Switch all blocks in the model to inference mode.
        """
        for m in self.modules():
            if isinstance(m, RepNeXtBlock):
                m.switch_to_deploy()
