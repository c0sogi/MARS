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
    RepVGG Block:
    - Training: 3x3 Conv + 1x1 Conv + Identity (multi-branch).
    - Inference: Fused 3x3 Conv (single-branch).
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
            )
        else:
            # 1. 3x3 Branch
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )

            # 2. 1x1 Branch
            # Only if kernel_size is 3, otherwise 1x1 doesn't make sense as a separate branch in this context
            assert kernel_size == 3
            padding_11 = padding - kernel_size // 2
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

            # 3. Identity Branch
            # Only if dimensions match and stride is 1
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

    def forward(self, inputs):
        if self.deploy:
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Derives the equivalent 3x3 kernel and bias by fusing all branches.
        """
        # Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # Fuse 1x1 branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        # Pad 1x1 kernel to 3x3
        kernel1x1_padded = self._pad_1x1_to_3x3_tensor(kernel1x1)

        # Fuse Identity branch
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        return kernel3x3 + kernel1x1_padded + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        # Pad (H, W) from 1x1 to 3x3.
        # Pad 1 pixel on all sides: (left, right, top, bottom) -> (1, 1, 1, 1)
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
                # Create an identity kernel for the convolution equivalent of the identity mapping
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
        if self.deploy:
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

        # Remove branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")

        self.deploy = True


class SelfEnsemblingRepVGG(nn.Module):
    """
    Self-Ensembling RepVGG Architecture.

    Structure:
    - Conservative Stem (3x3, stride 1)
    - Stage 1 (32x32)
    - Stage 2 (16x16) -> Aux Head
    - Stage 3 (8x8) -> Main Head
    """

    def __init__(self, num_classes=1, width_multiplier=1.0, deploy=False):
        super(SelfEnsemblingRepVGG, self).__init__()

        self.deploy = deploy
        self.num_classes = num_classes

        # Width configurations
        # Base widths: [64, 64, 128, 256]
        w_stem = int(64 * width_multiplier)
        w_s1 = int(64 * width_multiplier)
        w_s2 = int(128 * width_multiplier)
        w_s3 = int(256 * width_multiplier)

        # 1. Conservative Stem
        # Stride 1 to preserve 32x32 resolution
        self.stem = RepVGGBlock(
            in_channels=3, out_channels=w_stem, kernel_size=3, stride=1, deploy=deploy
        )

        # 2. Stage 1 (32x32)
        # 3 blocks
        self.stage1 = nn.ModuleList(
            [
                RepVGGBlock(w_stem, w_s1, stride=1, deploy=deploy),
                RepVGGBlock(w_s1, w_s1, stride=1, deploy=deploy),
                RepVGGBlock(w_s1, w_s1, stride=1, deploy=deploy),
            ]
        )

        # 3. Stage 2 (16x16)
        # Downsample at start. 3 blocks.
        self.stage2 = nn.ModuleList(
            [
                RepVGGBlock(w_s1, w_s2, stride=2, deploy=deploy),
                RepVGGBlock(w_s2, w_s2, stride=1, deploy=deploy),
                RepVGGBlock(w_s2, w_s2, stride=1, deploy=deploy),
            ]
        )

        # Auxiliary Head (attached after Stage 2)
        self.aux_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w_s2, num_classes)
        )

        # 4. Stage 3 (8x8)
        # Downsample at start. 3 blocks.
        self.stage3 = nn.ModuleList(
            [
                RepVGGBlock(w_s2, w_s3, stride=2, deploy=deploy),
                RepVGGBlock(w_s3, w_s3, stride=1, deploy=deploy),
                RepVGGBlock(w_s3, w_s3, stride=1, deploy=deploy),
            ]
        )

        # Main Head (attached after Stage 3)
        self.main_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(w_s3, num_classes)
        )

    def forward(self, x):
        # Stem
        out = self.stem(x)

        # Stage 1
        for block in self.stage1:
            out = block(out)

        # Stage 2
        for block in self.stage2:
            out = block(out)

        # Aux Output
        aux_out = self.aux_head(out)

        # Stage 3
        for block in self.stage3:
            out = block(out)

        # Main Output
        main_out = self.main_head(out)

        if self.training:
            # Return both for Deep Supervision Loss
            return main_out, aux_out
        else:
            # Internal Ensemble: Average predictions
            # Note: We average logits here. Since sigmoid is monotonic,
            # averaging logits is a valid ensemble strategy, though averaging probs is also common.
            # Given BCEWithLogitsLoss, logits are expected.
            return (main_out + aux_out) / 2.0

    def switch_to_deploy(self):
        """
        Recursively switches all RepVGG blocks to deployment mode (fused kernels).
        """
        if self.deploy:
            return

        self.stem.switch_to_deploy()
        for block in self.stage1:
            block.switch_to_deploy()
        for block in self.stage2:
            block.switch_to_deploy()
        for block in self.stage3:
            block.switch_to_deploy()

        self.deploy = True
