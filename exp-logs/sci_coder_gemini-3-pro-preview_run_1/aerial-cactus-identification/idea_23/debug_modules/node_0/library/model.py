import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from library.config import Config


class RepVGGBlock(nn.Module):
    """
    RepVGG Block that supports multi-branch training and structural re-parameterization for inference.
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
            # Identity branch (only if dimensions match)
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            # 3x3 Branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

            # 1x1 Branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=out_channels),
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
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
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
            in_channels=self.rbr_dense[0].in_channels,
            out_channels=self.rbr_dense[0].out_channels,
            kernel_size=self.rbr_dense[0].kernel_size,
            stride=self.rbr_dense[0].stride,
            padding=self.rbr_dense[0].padding,
            dilation=self.rbr_dense[0].dilation,
            groups=self.rbr_dense[0].groups,
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


class QualityRepVGG(nn.Module):
    """
    Quality-Supervised RepVGG Architecture.
    Includes an auxiliary head for file-size regression to enforce quality-aware feature learning.
    """

    def __init__(
        self,
        num_classes=1,
        width_multiplier=None,
        override_groups_map=None,
        deploy=False,
    ):
        super(QualityRepVGG, self).__init__()

        # Use Config defaults if not provided
        if width_multiplier is None:
            width_multiplier = Config.WIDTH_MULTIPLIER

        self.deploy = deploy
        self.override_groups_map = override_groups_map or dict()
        self.width_multiplier = width_multiplier

        # -- Architecture Definition for 32x32 Input --

        # Stage 0: Conservative Stem (32x32 -> 32x32)
        # Strictly 3x3 conv, stride 1 to preserve resolution
        self.in_planes = min(64, int(64 * width_multiplier))
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )
        self.cur_layer_idx = 1

        # Stage 1: High Res Feature Extraction (32x32 -> 32x32)
        self.stage1 = self._make_stage(
            int(64 * width_multiplier), num_blocks=2, stride=1
        )

        # Stage 2: Downsampling (32x32 -> 16x16)
        self.stage2 = self._make_stage(
            int(128 * width_multiplier), num_blocks=2, stride=2
        )

        # Stage 3: Downsampling (16x16 -> 8x8)
        self.stage3 = self._make_stage(
            int(256 * width_multiplier), num_blocks=2, stride=2
        )

        # Stage 4: Downsampling (8x8 -> 4x4)
        self.stage4 = self._make_stage(
            int(512 * width_multiplier), num_blocks=2, stride=2
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # -- Heads --
        final_dim = int(512 * width_multiplier)

        # 1. Classification Head
        self.linear_cls = nn.Linear(final_dim, num_classes)

        # 2. Auxiliary Quality Regression Head (File Size Prediction)
        # Only initialized if not in deploy mode. Removed during reparameterization.
        if not deploy:
            self.linear_qual = nn.Linear(final_dim, 1)
        else:
            self.linear_qual = None

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

        # Classification Output
        cls_out = self.linear_cls(out)

        # If deployed, return only classification logits
        if self.deploy:
            return cls_out

        # During training (or pre-deploy validation), return both
        if hasattr(self, "linear_qual") and self.linear_qual is not None:
            qual_out = self.linear_qual(out)
            return cls_out, qual_out
        else:
            # Fallback for safety
            return cls_out

    def reparameterize(self):
        """
        Converts the model to inference mode:
        1. Fuses all RepVGG blocks into single conv layers.
        2. Removes the auxiliary quality regression head.
        """
        # Fuse blocks
        for module in self.modules():
            if isinstance(module, RepVGGBlock):
                module.switch_to_deploy()

        # Remove auxiliary head
        if hasattr(self, "linear_qual"):
            self.linear_qual = None

        self.deploy = True
