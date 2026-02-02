import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class RepVGGBlock(nn.Module):
    """
    RepVGG Block:
    - Training: 3x3 Conv + 1x1 Conv + Identity (all with BN)
    - Inference: Fused 3x3 Conv
    """

    def __init__(
        self,
        in_channels,
        out_channels,
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
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        # Activation
        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )
        else:
            # Branch 1: Identity (only if dimensions match)
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # Branch 2: 3x3 Conv + BN
            self.rbr_dense = self.conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=groups,
            )

            # Branch 3: 1x1 Conv + BN
            self.rbr_1x1 = self.conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )

    def conv_bn(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
    ):
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

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        # Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # Fuse 1x1 branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        # Fuse Identity branch
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # Pad 1x1 kernel to 3x3
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
            # It's a Conv + BN branch
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # It's an Identity BN branch
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

        # Remove the training branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.deploy = True


class CustomRepVGG(nn.Module):
    """
    Custom RepVGG architecture optimized for 32x32 input.
    Features a conservative stem (stride 1) and 3x3 kernels exclusively.
    """

    def __init__(self, num_classes=1, width_multiplier=None, deploy=False):
        super(CustomRepVGG, self).__init__()

        # Default width multiplier configuration if not provided
        # Base channels: [64, 64, 128, 256, 512]
        if width_multiplier is None:
            # Using a relatively wide configuration for capacity
            self.stages_planes = [64, 64, 128, 256, 512]
        else:
            base = [64, 64, 128, 256, 512]
            self.stages_planes = [int(x * width_multiplier) for x in base]

        self.deploy = deploy
        self.in_planes = min(64, self.stages_planes[0])

        # Stage 0: Conservative Stem
        # 3x3 Conv, Stride 1, No pooling to preserve 32x32 resolution
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.stages_planes[0],
            stride=1,
            padding=1,
            deploy=deploy,
        )
        self.in_planes = self.stages_planes[0]

        # Stage 1: Resolution 32x32
        self.stage1 = self._make_stage(
            self.stages_planes[1], num_blocks=2, stride=1, deploy=deploy
        )

        # Stage 2: Resolution 16x16
        self.stage2 = self._make_stage(
            self.stages_planes[2], num_blocks=2, stride=2, deploy=deploy
        )

        # Stage 3: Resolution 8x8
        self.stage3 = self._make_stage(
            self.stages_planes[3], num_blocks=2, stride=2, deploy=deploy
        )

        # Stage 4: Resolution 4x4
        self.stage4 = self._make_stage(
            self.stages_planes[4], num_blocks=2, stride=2, deploy=deploy
        )

        # Classification Head
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(self.stages_planes[4], num_classes)

    def _make_stage(self, planes, num_blocks, stride, deploy):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    stride=s,
                    padding=1,
                    deploy=deploy,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def switch_to_deploy(self):
        """
        Recursively switches all RepVGG blocks to deploy mode (fusing branches).
        """
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True
