import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from library.config import STAGES_CHANNELS, CARDINALITY, USE_SE, NUM_CLASSES
from library.utils import fuse_conv_bn, pad_1x1_to_3x3_tensor


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    """

    def __init__(self, input_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure reduction doesn't make channels too small
        reduced_channels = max(input_channels // reduction_ratio, 8)
        self.fc1 = nn.Conv2d(input_channels, reduced_channels, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, input_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return x * scale


class RepNeXtBlock(nn.Module):
    """
    RepNeXt Block:
    Training: Parallel Grouped 3x3, Grouped 1x1, Identity (if shapes match).
    Inference: Fused Grouped 3x3.
    """

    def __init__(
        self, in_channels, out_channels, stride=1, groups=32, deploy=False, use_se=True
    ):
        super(RepNeXtBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_se = use_se

        # Padding for 3x3 to maintain size
        padding = 1

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
            )
        else:
            # Branch 1: Grouped 3x3
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # Branch 2: Grouped 1x1
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

            # Branch 3: Identity
            # Only if dimensions match and stride is 1
            self.rbr_identity = None
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(in_channels)

        self.non_linearity = nn.ReLU(inplace=True)
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

    def forward(self, x):
        if self.deploy:
            return self.non_linearity(self.se(self.rbr_reparam(x)))

        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)

        return self.non_linearity(self.se(out))

    def switch_to_deploy(self):
        if self.deploy:
            return

        # Fuse 3x3
        kernel_3x3, bias_3x3 = fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])

        # Fuse 1x1
        kernel_1x1, bias_1x1 = fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        kernel_1x1 = pad_1x1_to_3x3_tensor(kernel_1x1)

        # Fuse Identity
        kernel_id = 0
        bias_id = 0
        if self.rbr_identity is not None:
            # Create identity kernel for grouped conv
            # Shape: (C, C/g, 3, 3) after padding (originally 1x1)
            input_dim = self.in_channels // self.groups
            kernel_value = np.zeros(
                (self.in_channels, input_dim, 3, 3), dtype=np.float32
            )
            for i in range(self.in_channels):
                # The input index within the group corresponding to output i is i % input_dim
                kernel_value[i, i % input_dim, 1, 1] = 1

            id_tensor = torch.from_numpy(kernel_value).to(
                self.rbr_identity.weight.device
            )

            # Fuse with BN
            running_mean = self.rbr_identity.running_mean
            running_var = self.rbr_identity.running_var
            gamma = self.rbr_identity.weight
            beta = self.rbr_identity.bias
            eps = self.rbr_identity.eps

            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            kernel_id = id_tensor * t
            bias_id = beta - running_mean * gamma / std

        # Sum everything
        final_kernel = kernel_3x3 + kernel_1x1 + kernel_id
        final_bias = bias_3x3 + bias_1x1 + bias_id

        # Create new layer
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = final_kernel
        self.rbr_reparam.bias.data = final_bias

        # Remove old branches
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")

        self.deploy = True


class RepDownsample(nn.Module):
    """
    Downsampling block: Stride 2.
    Parallel 3x3 (stride 2) and 1x1 (stride 2).
    """

    def __init__(self, in_channels, out_channels, groups=32, deploy=False):
        super(RepDownsample, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=2,
                    padding=0,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.deploy:
            return self.non_linearity(self.rbr_reparam(x))
        return self.non_linearity(self.rbr_dense(x) + self.rbr_1x1(x))

    def switch_to_deploy(self):
        if self.deploy:
            return

        kernel_3x3, bias_3x3 = fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        kernel_1x1, bias_1x1 = fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        kernel_1x1 = pad_1x1_to_3x3_tensor(kernel_1x1)

        final_kernel = kernel_3x3 + kernel_1x1
        final_bias = bias_3x3 + bias_1x1

        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = final_kernel
        self.rbr_reparam.bias.data = final_bias

        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        self.deploy = True


class UltraWideRepResNeXt(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, deploy=False):
        super(UltraWideRepResNeXt, self).__init__()
        self.deploy = deploy

        # Configs
        widths = STAGES_CHANNELS  # [96, 192, 384]
        groups = CARDINALITY  # 32

        # Stem: 3 -> 96. Dense RepConv.
        # Since input is 3 channels, we can't use groups=32. We use groups=1 (Dense).
        self.stem = RepNeXtBlock(
            3, widths[0], stride=1, groups=1, deploy=deploy, use_se=False
        )

        # Stage 1: 96 channels. 32x32.
        self.stage1 = nn.Sequential(
            RepNeXtBlock(
                widths[0], widths[0], groups=groups, deploy=deploy, use_se=USE_SE
            ),
            RepNeXtBlock(
                widths[0], widths[0], groups=groups, deploy=deploy, use_se=USE_SE
            ),
        )

        # Downsample 1: 96 -> 192. 32x32 -> 16x16.
        self.down1 = RepDownsample(widths[0], widths[1], groups=groups, deploy=deploy)

        # Stage 2: 192 channels. 16x16.
        self.stage2 = nn.Sequential(
            RepNeXtBlock(
                widths[1], widths[1], groups=groups, deploy=deploy, use_se=USE_SE
            ),
            RepNeXtBlock(
                widths[1], widths[1], groups=groups, deploy=deploy, use_se=USE_SE
            ),
        )

        # Downsample 2: 192 -> 384. 16x16 -> 8x8.
        self.down2 = RepDownsample(widths[1], widths[2], groups=groups, deploy=deploy)

        # Stage 3: 384 channels. 8x8.
        self.stage3 = nn.Sequential(
            RepNeXtBlock(
                widths[2], widths[2], groups=groups, deploy=deploy, use_se=USE_SE
            ),
            RepNeXtBlock(
                widths[2], widths[2], groups=groups, deploy=deploy, use_se=USE_SE
            ),
        )

        # Head
        # Concat GAP(Stage2) + GAP(Stage3)
        # Stage 2 dim: 192. Stage 3 dim: 384. Total: 576.
        self.final_dim = widths[1] + widths[2]
        self.classifier = nn.Linear(self.final_dim, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        # Stage 2 path
        x = self.down1(x)
        x = self.stage2(x)
        feat_s2 = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Stage 3 path
        x = self.down2(x)
        x = self.stage3(x)
        feat_s3 = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # Aggregation
        feat = torch.cat([feat_s2, feat_s3], dim=1)
        logits = self.classifier(feat)
        return logits

    def switch_to_deploy(self):
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()
        self.deploy = True
