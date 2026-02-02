import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper to create a Conv2d followed by a BatchNorm2d.
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
    RepVGG Block that supports Structural Re-parameterization.

    Training: 3x3 Branch + 1x1 Branch + Identity Branch
    Inference: Fused 3x3 Branch
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # Standard ReLU activation
        self.nonlinearity = nn.ReLU()

        if deploy:
            # In deploy mode, we only need a single conv
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            # Training mode: Multi-branch topology

            # 1. 3x3 Branch (Dense)
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )

            # 2. 1x1 Branch
            # Padding is 0 for 1x1 conv to maintain spatial dimensions
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )

            # 3. Identity Branch
            # Only exists if input/output dimensions match and stride is 1
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = None

    def forward(self, inputs):
        # Deploy mode
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        # Training mode
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculates the equivalent 3x3 kernel and bias by fusing all branches.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # Pad 1x1 kernel to 3x3 and sum everything
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            # Pad H and W dimensions by 1 on each side
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            # Branch is Conv + BN
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # Branch is BN only (Identity)
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                # Create an identity convolution kernel
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        # Fusion math: W' = W * (gamma / std), B' = beta - mean * (gamma / std)
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        """
        Converts the block to inference mode by fusing branches.
        """
        if hasattr(self, "rbr_reparam"):
            return

        kernel, bias = self.get_equivalent_kernel_bias()

        # Create the fused convolution
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

        # Delete training branches to free memory
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.deploy = True


class CactusRepVGG(nn.Module):
    """
    Custom RepVGG Architecture for Cactus Identification.

    Features:
    - Conservative Stem (32x32 preserved)
    - Dual Heads (Texture and Semantic)
    - Structural Re-parameterization
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, width_multiplier=1.0):
        super(CactusRepVGG, self).__init__()

        # Define channel widths
        self.in_planes = min(64, int(64 * width_multiplier))
        w1 = int(64 * width_multiplier)
        w2 = int(128 * width_multiplier)
        w3 = int(256 * width_multiplier)

        # Stem: 3x3, Stride 1. Preserves 32x32 resolution.
        self.stem = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Stage 1: 32x32 -> 16x16
        self.stage1 = self._make_stage(planes=w1, num_blocks=2, stride=2)

        # Stage 2: 16x16 -> 8x8
        self.stage2 = self._make_stage(planes=w2, num_blocks=3, stride=2)

        # Stage 3: 8x8 -> 4x4
        self.stage3 = self._make_stage(planes=w3, num_blocks=3, stride=2)

        # Pooling
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Permanent Heads
        # Texture Head: Attached after Stage 2 (8x8)
        self.texture_head = nn.Linear(w2, num_classes)

        # Semantic Head: Attached after Stage 3 (4x4)
        self.semantic_head = nn.Linear(w3, num_classes)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=s,
                    padding=1,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Stem
        out = self.stem(x)

        # Stage 1
        out = self.stage1(out)

        # Stage 2
        out = self.stage2(out)
        # Extract Texture Features
        feat_texture = self.gap(out).view(out.size(0), -1)
        logits_texture = self.texture_head(feat_texture)

        # Stage 3
        out = self.stage3(out)
        # Extract Semantic Features
        feat_semantic = self.gap(out).view(out.size(0), -1)
        logits_semantic = self.semantic_head(feat_semantic)

        return logits_texture, logits_semantic

    def switch_to_deploy(self):
        """
        Recursively switches all RepVGG blocks to deployment mode.
        """
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
