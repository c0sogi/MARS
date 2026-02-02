import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RepNeXtBlock(nn.Module):
    """
    RepNeXt Block:
    - Training: Parallel 3x3, 1x1, and Identity branches.
    - Inference: Fused into a single 3x3 convolution.
    - Supports Grouped Convolutions (RepNeXt).
    - Supports Downsampling (Stride > 1).
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=1, use_se=True):
        super(RepNeXtBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.use_se = use_se
        self.deploy = False

        # Ensure groups is valid for the given channels
        # If channels are not divisible by groups (e.g. input layer), force groups=1
        if in_channels % groups != 0 or out_channels % groups != 0:
            self.groups = 1

        # Padding for 3x3 conv to maintain spatial dimensions (if stride=1)
        padding = 1

        if self.deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=self.groups,
                bias=True,
            )
        else:
            # 3x3 Branch
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    groups=self.groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # 1x1 Branch
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=self.groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # Identity Branch
            # Only if input/output dimensions match and stride is 1
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)
            else:
                self.rbr_identity = None

        # Activation
        self.act = nn.ReLU(inplace=True)

        # SE Block
        if self.use_se:
            self.se = SEBlock(out_channels, reduction=Config.SE_REDUCTION)

    def forward(self, x):
        if self.deploy:
            out = self.rbr_reparam(x)
        else:
            # Sum of branches
            x_dense = self.rbr_dense(x)
            x_1x1 = self.rbr_1x1(x)
            out = x_dense + x_1x1

            if self.rbr_identity is not None:
                out += self.rbr_identity(x)

        # Apply SE before activation (or after, depending on design, here we do before ReLU logic usually,
        # but RepVGG does ReLU on the sum. We apply SE on the sum, then ReLU)
        if self.use_se:
            out = self.se(out)

        return self.act(out)

    def get_equivalent_kernel_bias(self):
        # 1. Fuse 3x3 branch
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)

        # 2. Fuse 1x1 branch
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)

        # Pad 1x1 kernel to 3x3
        # kernel1x1 is (out, in/g, 1, 1). Pad to (out, in/g, 3, 3) with 1 on each side
        kernel1x1_padded = F.pad(kernel1x1, [1, 1, 1, 1])

        # 3. Fuse Identity branch
        kernel_id, bias_id = 0, 0
        if self.rbr_identity is not None:
            # Identity kernel: 1s on the corresponding channel positions
            # Shape: (out, in/g, 3, 3)
            kernel_id_val = torch.zeros(
                self.out_channels, self.in_channels // self.groups, 3, 3
            ).to(kernel3x3.device)
            for i in range(self.out_channels):
                # Determine the input channel index within the group
                # For grouped conv, the input dim is in_channels // groups
                # The filter i connects to input channel i (since it's identity)
                # But we need to map global channel index to group-relative index
                # Group index: i // (out_c // groups)
                # Input channel index is just i % (in_c // groups) if in==out
                kernel_id_val[i, i % (self.in_channels // self.groups), 1, 1] = 1

            kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity, kernel_id_val)

        return kernel3x3 + kernel1x1_padded + kernel_id, bias3x3 + bias1x1 + bias_id

    def _fuse_bn_tensor(self, branch, kernel_init=None):
        if isinstance(branch, nn.Sequential):
            # Conv + BN
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            # Just BN (Identity branch)
            assert isinstance(branch, nn.BatchNorm2d)
            if kernel_init is None:
                raise ValueError("Kernel init required for BN-only branch")
            kernel = kernel_init
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
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # Remove training branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True


class UltraWideSERepNeXt(nn.Module):
    """
    Custom Ultra-Wide SE-RepNeXt Architecture.
    - 3 Stages with channels [96, 192, 384].
    - Multi-Scale Aggregation (Stage 2 + Stage 3).
    """

    def __init__(self, num_blocks=[3, 3, 3]):
        super(UltraWideSERepNeXt, self).__init__()

        self.stage_channels = Config.STAGE_CHANNELS  # [96, 192, 384]
        self.groups = Config.GROUPS

        # --- Stem ---
        # 3 -> 96. Stride 1. Groups=1 (since 3 < 32).
        self.stem = RepNeXtBlock(
            3, self.stage_channels[0], stride=1, groups=1, use_se=Config.USE_SE
        )

        # --- Stage 1 ---
        # 96 -> 96. Stride 1. Groups=32.
        stage1_blocks = []
        for _ in range(num_blocks[0]):
            stage1_blocks.append(
                RepNeXtBlock(
                    self.stage_channels[0],
                    self.stage_channels[0],
                    stride=1,
                    groups=self.groups,
                    use_se=Config.USE_SE,
                )
            )
        self.stage1 = nn.Sequential(*stage1_blocks)

        # --- Stage 2 ---
        # 96 -> 192. First block stride 2 (Downsample). Groups=32.
        stage2_blocks = []
        # Downsampling block
        stage2_blocks.append(
            RepNeXtBlock(
                self.stage_channels[0],
                self.stage_channels[1],
                stride=2,
                groups=self.groups,
                use_se=Config.USE_SE,
            )
        )
        # Normal blocks
        for _ in range(num_blocks[1] - 1):
            stage2_blocks.append(
                RepNeXtBlock(
                    self.stage_channels[1],
                    self.stage_channels[1],
                    stride=1,
                    groups=self.groups,
                    use_se=Config.USE_SE,
                )
            )
        self.stage2 = nn.Sequential(*stage2_blocks)

        # --- Stage 3 ---
        # 192 -> 384. First block stride 2 (Downsample). Groups=32.
        stage3_blocks = []
        # Downsampling block
        stage3_blocks.append(
            RepNeXtBlock(
                self.stage_channels[1],
                self.stage_channels[2],
                stride=2,
                groups=self.groups,
                use_se=Config.USE_SE,
            )
        )
        # Normal blocks
        for _ in range(num_blocks[2] - 1):
            stage3_blocks.append(
                RepNeXtBlock(
                    self.stage_channels[2],
                    self.stage_channels[2],
                    stride=1,
                    groups=self.groups,
                    use_se=Config.USE_SE,
                )
            )
        self.stage3 = nn.Sequential(*stage3_blocks)

        # --- Head ---
        # Multi-Scale Aggregation: GAP(Stage2) + GAP(Stage3)
        # Stage 2 out: 192 channels. Stage 3 out: 384 channels.
        # Total features = 192 + 384 = 576.
        self.final_channels = self.stage_channels[1] + self.stage_channels[2]
        self.classifier = nn.Linear(self.final_channels, Config.NUM_CLASSES)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Stage 1 (32x32)
        x = self.stage1(x)

        # Stage 2 (16x16)
        x2 = self.stage2(x)

        # Stage 3 (8x8)
        x3 = self.stage3(x2)

        # Multi-Scale Aggregation
        # GAP on Stage 2 features
        feat2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        # GAP on Stage 3 features
        feat3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)

        # Concatenate
        combined = torch.cat([feat2, feat3], dim=1)

        # Classification
        out = self.classifier(combined)
        return out

    def switch_to_deploy(self):
        """
        Recursively switches all RepNeXtBlocks to deploy mode.
        """
        for m in self.modules():
            if isinstance(m, RepNeXtBlock):
                m.switch_to_deploy()
