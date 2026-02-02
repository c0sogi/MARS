import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper function to create a Convolution + BatchNorm sequential module.
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
    During training: 3x3 Conv + 1x1 Conv + Identity.
    During inference: Fused 3x3 Conv.
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

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.non_linearity = nn.ReLU()

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
                padding=padding_11,
                groups=groups,
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


class MultiTaskRepVGG(nn.Module):
    def __init__(self, deploy=False):
        super(MultiTaskRepVGG, self).__init__()
        self.deploy = deploy

        # Width Multiplier from Config
        width_mult = Config.BACKBONE_WIDTH_MULTIPLIER

        # Define channel widths (Custom small architecture for 32x32)
        # Stage 0 (Stem): 32x32 resolution
        # Stage 1: 32x32 resolution
        # Stage 2: 16x16 resolution
        # Stage 3: 8x8 resolution

        self.in_planes = min(64, int(64 * width_mult))

        # Stage 0: Stem (Conservative, stride 1)
        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )

        # Stage 1
        planes_s1 = int(64 * width_mult)
        self.stage1 = self._make_stage(planes_s1, num_blocks=2, stride=1, deploy=deploy)

        # Stage 2
        planes_s2 = int(128 * width_mult)
        self.stage2 = self._make_stage(planes_s2, num_blocks=3, stride=2, deploy=deploy)

        # Stage 3
        planes_s3 = int(256 * width_mult)
        self.stage3 = self._make_stage(planes_s3, num_blocks=4, stride=2, deploy=deploy)

        # Pooling
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # --- Heads ---

        # Texture Head (Attached to Stage 2 Output)
        self.use_texture = Config.USE_TEXTURE_HEAD
        if self.use_texture:
            self.head_texture = nn.Linear(planes_s2, 1)

        # Semantic Head (Attached to Stage 3 Output)
        self.use_semantic = Config.USE_SEMANTIC_HEAD
        if self.use_semantic:
            self.head_semantic = nn.Linear(planes_s3, 1)

        # Quality Head (Attached to Stage 3 Output - Regression)
        self.use_quality = Config.USE_QUALITY_HEAD
        if self.use_quality:
            self.head_quality = nn.Linear(planes_s3, 1)

    def _make_stage(self, planes, num_blocks, stride, deploy):
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
                    deploy=deploy,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        # Backbone Forward
        out = self.stage0(x)
        out = self.stage1(out)

        # Stage 2 Output (16x16)
        out_s2 = self.stage2(out)

        # Texture Head Prediction
        texture_logits = None
        if self.use_texture:
            feat_s2 = self.gap(out_s2).view(out_s2.size(0), -1)
            texture_logits = self.head_texture(feat_s2)

        # Stage 3 Output (8x8)
        out_s3 = self.stage3(out_s2)
        feat_s3 = self.gap(out_s3).view(out_s3.size(0), -1)

        # Semantic Head Prediction
        semantic_logits = None
        if self.use_semantic:
            semantic_logits = self.head_semantic(feat_s3)

        # Quality Head Prediction (Regression)
        quality_pred = None
        if self.use_quality:
            quality_pred = self.head_quality(feat_s3)

        return {
            "texture": texture_logits,
            "semantic": semantic_logits,
            "quality": quality_pred,
        }

    def reparameterize(self):
        """
        Iterates through all modules and switches RepVGGBlocks to deploy mode.
        This permanently fuses branches for faster inference.
        """
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()
        self.deploy = True
