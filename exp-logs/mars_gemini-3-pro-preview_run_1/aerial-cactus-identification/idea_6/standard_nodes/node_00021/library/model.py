import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class RepVGGBlock(nn.Module):
    """
    RepVGG Block that supports structural re-parameterization.
    Training: Multi-branch (3x3, 1x1, Identity).
    Inference: Fused 3x3 Conv.
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
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.padding_mode = padding_mode

        # Activation
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
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding - kernel_size // 2,
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
            # Identity branch is just BN
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


class CactusRepVGG(nn.Module):
    """
    Simplified RepVGG architecture without Deep Supervision.
    Structure: Stem -> Stage1 -> Stage2 -> Stage3 -> MainHead.
    """

    def __init__(self, num_classes=1, width_multiplier=1.0, deploy=False):
        super(CactusRepVGG, self).__init__()
        self.deploy = deploy
        self.num_classes = num_classes

        # Channel configurations
        self.stage_planes = [
            int(64 * width_multiplier),
            int(64 * width_multiplier),
            int(128 * width_multiplier),
            int(256 * width_multiplier),
        ]

        # Stem: 3x3 Conv, Stride 1 (Preserves 32x32 resolution)
        self.stem = RepVGGBlock(
            in_channels=3,
            out_channels=self.stage_planes[0],
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )

        # Stages
        # Stage 1: 32x32 -> 16x16
        self.stage1 = self._make_stage(
            self.stage_planes[0],
            self.stage_planes[1],
            num_blocks=2,
            stride=2,
            deploy=deploy,
        )

        # Stage 2: 16x16 -> 8x8
        self.stage2 = self._make_stage(
            self.stage_planes[1],
            self.stage_planes[2],
            num_blocks=2,
            stride=2,
            deploy=deploy,
        )

        # Stage 3: 8x8 -> 4x4
        self.stage3 = self._make_stage(
            self.stage_planes[2],
            self.stage_planes[3],
            num_blocks=2,
            stride=2,
            deploy=deploy,
        )

        # Main Head attached after Stage 3
        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.stage_planes[3], num_classes),
        )

    def _make_stage(self, in_planes, out_planes, num_blocks, stride, deploy):
        layers = []
        layers.append(
            RepVGGBlock(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                deploy=deploy,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                RepVGGBlock(
                    in_channels=out_planes,
                    out_channels=out_planes,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    deploy=deploy,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        return self.main_head(out)

    def switch_to_deploy(self):
        """
        Fuses all RepVGG blocks for efficient inference.
        """
        if self.deploy:
            return

        self.stem.switch_to_deploy()
        for stage in [self.stage1, self.stage2, self.stage3]:
            for layer in stage:
                layer.switch_to_deploy()

        self.deploy = True
