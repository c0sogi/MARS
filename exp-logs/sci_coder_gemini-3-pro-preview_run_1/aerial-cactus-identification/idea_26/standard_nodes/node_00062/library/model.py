import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


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
    RepVGG Block that supports structural reparameterization.

    During training: 3x3 Conv + 1x1 Conv + Identity (if applicable).
    During inference: Fused into a single 3x3 Conv.
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
        self.padding_mode = padding_mode

        # Activation
        self.nonlinearity = nn.ReLU()

        if deploy:
            # Inference mode: Single Conv layer
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
            # Training mode: Multi-branch
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
        """
        Calculates the equivalent 3x3 kernel and bias by fusing all branches.
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
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
            # Conv + BN
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # Identity Branch (Just BN)
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
        """
        Converts the multi-branch structure into a single convolution for inference.
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

        # Remove training branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True


class QualityRepVGG(nn.Module):
    """
    Quality-Supervised RepVGG Architecture.

    Features:
    - Conservative Stem (Stride 1) to preserve 32x32 resolution.
    - Multi-task learning: Classification + Quality (File Size) Regression.
    """

    def __init__(
        self,
        num_blocks,
        num_classes=1,
        width_multiplier=None,
        override_groups_map=None,
        deploy=False,
    ):
        super(QualityRepVGG, self).__init__()

        # Default width multipliers for RepVGG-A0 like structure if not provided
        if width_multiplier is None:
            width_multiplier = [0.75, 0.75, 0.75, 2.5]

        assert len(width_multiplier) == 4
        assert len(num_blocks) == 4

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()

        # Base widths
        self.in_planes = min(64, int(64 * width_multiplier[0]))

        # Stage 0: Conservative Stem
        # Strictly 3x3, Stride 1 to preserve 32x32 resolution
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )
        self.cur_layer_idx = 1

        # Stage 1
        self.stage1 = self._make_stage(
            int(64 * width_multiplier[0]), num_blocks[0], stride=1
        )  # Keep 32x32

        # Stage 2
        self.stage2 = self._make_stage(
            int(128 * width_multiplier[1]), num_blocks[1], stride=2
        )  # 16x16

        # Stage 3
        self.stage3 = self._make_stage(
            int(256 * width_multiplier[2]), num_blocks[2], stride=2
        )  # 8x8

        # Stage 4
        self.stage4 = self._make_stage(
            int(512 * width_multiplier[3]), num_blocks[3], stride=2
        )  # 4x4

        # Heads
        final_channels = int(512 * width_multiplier[3])
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Classification Head
        self.linear_cls = nn.Linear(final_channels, num_classes)

        # Auxiliary Quality Head (Regression)
        self.linear_qual = nn.Linear(final_channels, 1)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(
                RepVGGBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=3,
                    stride=s,
                    padding=1,
                    groups=cur_groups,
                    deploy=self.deploy,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)

        cls_out = self.linear_cls(out)
        qual_out = self.linear_qual(out)

        return cls_out, qual_out

    def reparameterize_model(self):
        """
        Recursively switches all RepVGGBlocks to deploy mode.
        """
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True


def get_repvgg_model(model_name="RepVGG-A0", deploy=False):
    """
    Factory function to get the model.
    """
    # Configs adapted for small input size (lighter width multipliers)
    # RepVGG-A0 original: [0.75, 0.75, 0.75, 2.5]

    if model_name == "RepVGG-A0":
        return QualityRepVGG(
            num_blocks=[1, 2, 4, 1],
            width_multiplier=[0.75, 0.75, 0.75, 2.5],
            deploy=deploy,
        )
    elif model_name == "RepVGG-A1":
        return QualityRepVGG(
            num_blocks=[1, 2, 4, 1], width_multiplier=[1, 1, 1, 2.5], deploy=deploy
        )
    elif model_name == "RepVGG-A2":
        return QualityRepVGG(
            num_blocks=[1, 2, 4, 1],
            width_multiplier=[1.5, 1.5, 1.5, 2.75],
            deploy=deploy,
        )
    else:
        # Default lightweight
        return QualityRepVGG(
            num_blocks=[1, 2, 2, 1],
            width_multiplier=[0.5, 0.5, 0.5, 1.0],
            deploy=deploy,
        )
