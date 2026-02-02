import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import NUM_CLASSES


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper function to create a Conv2d followed by BatchNorm2d.
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

    Training: Multi-branch (3x3, 1x1, Identity).
    Inference: Single-branch (Fused 3x3).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
    ):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert kernel_size == 3
        assert padding == 1

        self.nonlinearity = nn.ReLU()

        if deploy:
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
            # Identity branch exists only if dimensions match and stride is 1
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            # 3x3 Branch
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            # 1x1 Branch
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # Identity branch is just BatchNorm
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
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

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
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
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


class MultiScaleRepVGG(nn.Module):
    """
    Multi-Scale Self-Ensembling RepVGG.

    Features:
    - Conservative Stem (Stride 1) for 32x32 images.
    - 3 Classification Heads (Early, Middle, Late).
    - RepVGG Blocks for efficient inference.
    """

    def __init__(self, num_classes=NUM_CLASSES, deploy=False):
        super(MultiScaleRepVGG, self).__init__()
        self.deploy = deploy

        # Stem: 3 -> 64, Stride 1 (Preserve 32x32)
        self.stem = RepVGGBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )

        # Stage 1: 3 Blocks, 64 Channels, Stride 1. Output: 32x32
        self.stage1_layers = nn.ModuleList(
            [RepVGGBlock(64, 64, stride=1, deploy=deploy) for _ in range(3)]
        )

        # Head 1 (Early)
        self.head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes)
        )

        # Stage 2: Downsample + 2 Blocks. 64 -> 128. Output: 16x16
        self.stage2_layers = nn.ModuleList()
        self.stage2_layers.append(RepVGGBlock(64, 128, stride=2, deploy=deploy))
        for _ in range(2):
            self.stage2_layers.append(RepVGGBlock(128, 128, stride=1, deploy=deploy))

        # Head 2 (Middle)
        self.head2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, num_classes)
        )

        # Stage 3: Downsample + 2 Blocks. 128 -> 256. Output: 8x8
        self.stage3_layers = nn.ModuleList()
        self.stage3_layers.append(RepVGGBlock(128, 256, stride=2, deploy=deploy))
        for _ in range(2):
            self.stage3_layers.append(RepVGGBlock(256, 256, stride=1, deploy=deploy))

        # Head 3 (Late)
        self.head3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1
        for layer in self.stage1_layers:
            x = layer(x)
        out1 = self.head1(x)

        # Stage 2
        for layer in self.stage2_layers:
            x = layer(x)
        out2 = self.head2(x)

        # Stage 3
        for layer in self.stage3_layers:
            x = layer(x)
        out3 = self.head3(x)

        # Return all heads for self-ensembling loss and inference aggregation
        return [out1, out2, out3]


def reparameterize_model(model):
    """
    Converts a MultiScaleRepVGG model from training mode (multi-branch)
    to inference mode (single-branch fused).
    """
    # Recursively find all RepVGGBlocks and switch them
    for module in model.modules():
        if hasattr(module, "switch_to_deploy"):
            module.switch_to_deploy()
    model.deploy = True
    return model
