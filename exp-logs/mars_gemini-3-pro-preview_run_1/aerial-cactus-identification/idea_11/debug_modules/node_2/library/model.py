import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    """
    Helper to create a Conv2d followed by BatchNorm2d.
    Bias is set to False for the Conv2d because BN follows.
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
    - Training: 3x3 branch + 1x1 branch + Identity branch (if applicable).
    - Inference: Fused into a single 3x3 Conv2d.
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
            # Branch 1: 3x3 Conv + BN
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )

            # Branch 2: 1x1 Conv + BN
            # 1x1 conv padding is 0
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
            )

            # Branch 3: Identity + BN (only if dims match and stride is 1)
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = None

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
        Fuses the 3 branches into a single kernel and bias.
        """
        # Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # Fuse 1x1 branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        # Fuse Identity branch
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)

        # Combine
        # Pad 1x1 kernel to 3x3
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            # Pad (H, W) from 1x1 to 3x3 -> pad 1 on all sides
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0

        if isinstance(branch, nn.Sequential):
            # It's a Conv + BN block
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            # It's just a BN block (Identity branch)
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                # Construct an identity kernel
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

        # Delete training branches
        for para in self.parameters():
            para.detach_()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        if hasattr(self, "id_tensor"):
            del self.id_tensor

        self.deploy = True


class MetadataFusedRepVGG(nn.Module):
    """
    Custom RepVGG architecture for 32x32 images with Metadata Fusion.

    Structure:
    - Stem: 3x3 Conv (stride 1) - Preserves 32x32 resolution.
    - Stage 1: RepVGG Blocks (32x32)
    - Stage 2: RepVGG Blocks (16x16)
    - Stage 3: RepVGG Blocks (8x8)
    - Stage 4: RepVGG Blocks (4x4)
    - Head: GlobalAvgPool + Concat(Metadata) + Linear
    """

    def __init__(self, num_classes=1, deploy=False):
        super(MetadataFusedRepVGG, self).__init__()

        self.deploy = deploy

        # Widths for stages.
        # Base: 64. Multipliers: [1, 2, 4, 8]
        # Stage 0 (Stem): 64
        # Stage 1: 64
        # Stage 2: 128
        # Stage 3: 256
        # Stage 4: 512

        self.stage0 = RepVGGBlock(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            deploy=deploy,
        )

        self.stage1 = self._make_stage(64, 64, num_blocks=2, stride=1, deploy=deploy)
        self.stage2 = self._make_stage(64, 128, num_blocks=2, stride=2, deploy=deploy)
        self.stage3 = self._make_stage(128, 256, num_blocks=2, stride=2, deploy=deploy)
        self.stage4 = self._make_stage(256, 512, num_blocks=2, stride=2, deploy=deploy)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

        # Metadata Fusion Head
        # Visual features: 512
        # Metadata features: 1 (normalized file size)
        # Total input to linear: 513
        self.linear = nn.Linear(512 + 1, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride, deploy):
        layers = []
        # First block handles stride and channel change
        layers.append(
            RepVGGBlock(in_channels, out_channels, stride=stride, deploy=deploy)
        )
        # Subsequent blocks keep dimensions
        for _ in range(1, num_blocks):
            layers.append(
                RepVGGBlock(out_channels, out_channels, stride=1, deploy=deploy)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Tuple containing (image_tensor, metadata_tensor)
               image_tensor: (B, 3, 32, 32)
               metadata_tensor: (B) or (B, 1) - Normalized file size
        """
        img, meta = x

        # Ensure metadata is (B, 1)
        if meta.dim() == 1:
            meta = meta.view(-1, 1)

        # Backbone
        out = self.stage0(img)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # Global Pooling
        out = self.gap(out)
        out = out.view(out.size(0), -1)  # Flatten to (B, 512)

        # Metadata Fusion
        # Concatenate visual features with metadata
        fused = torch.cat([out, meta], dim=1)  # (B, 513)

        # Classification
        out = self.linear(fused)

        return out

    def switch_to_deploy(self):
        """
        Recursively switches all RepVGG blocks to deploy mode.
        """
        for m in self.modules():
            if isinstance(m, RepVGGBlock):
                m.switch_to_deploy()
        self.deploy = True
