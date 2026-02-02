import torch
import torch.nn as nn
import torch.nn.functional as F
from library.utils import fuse_conv_bn
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Performs channel-wise attention to recalibrate feature maps.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Ensure reduced channels is at least 1
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class RepNeXtBlock(nn.Module):
    """
    RepNeXt Block with multi-branch training and fused inference (Structural Re-parameterization).

    Training Topology:
        1. Grouped 3x3 Convolution
        2. Grouped 1x1 Convolution
        3. Identity Mapping (if input/output dimensions match)

    Inference Topology:
        1. Fused Grouped 3x3 Convolution

    Both topologies are followed by an SE Block and Activation.
    """

    def __init__(self, in_channels, out_channels, stride=1, groups=32, deploy=False):
        super(RepNeXtBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # SE Block is applied after the convolution (fused or summed)
        self.se = SEBlock(out_channels)
        self.activation = nn.ReLU(inplace=True)

        if deploy:
            # Inference: Single fused layer
            self.reparam_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=True,
            )
        else:
            # Training: Multi-branch topology

            # Branch 1: Grouped 3x3 Conv
            self.conv3x3 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            )
            self.bn3x3 = nn.BatchNorm2d(out_channels)

            # Branch 2: Grouped 1x1 Conv
            self.conv1x1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
                bias=False,
            )
            self.bn1x1 = nn.BatchNorm2d(out_channels)

            # Branch 3: Identity
            # Valid only if dimensions match and stride is 1
            if out_channels == in_channels and stride == 1:
                self.identity = nn.BatchNorm2d(out_channels)
            else:
                self.identity = None

    def forward(self, x):
        if self.deploy:
            y = self.reparam_conv(x)
            y = self.se(y)
            return self.activation(y)

        # Training forward: Sum of branches
        y = self.bn3x3(self.conv3x3(x))
        y += self.bn1x1(self.conv1x1(x))

        if self.identity is not None:
            y += self.identity(x)

        y = self.se(y)
        return self.activation(y)

    def switch_to_deploy(self):
        """
        Fuses the multi-branch topology into a single convolutional layer for inference.
        """
        if self.deploy:
            return

        # 1. Fuse 3x3 Branch
        fused_3x3 = fuse_conv_bn(self.conv3x3, self.bn3x3)
        w_3x3 = fused_3x3.weight
        b_3x3 = fused_3x3.bias

        # 2. Fuse 1x1 Branch
        fused_1x1 = fuse_conv_bn(self.conv1x1, self.bn1x1)
        # Pad 1x1 weights (Co, Ci/g, 1, 1) to 3x3 (Co, Ci/g, 3, 3)
        w_1x1 = F.pad(fused_1x1.weight, (1, 1, 1, 1))
        b_1x1 = fused_1x1.bias

        # 3. Fuse Identity Branch
        w_id = 0
        b_id = 0
        if self.identity is not None:
            # Create a dummy 1x1 grouped conv with identity weights
            dummy_conv = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                kernel_size=1,
                groups=self.groups,
                bias=False,
            )

            # Construct identity kernel for grouped convolution
            # Shape: (out_channels, in_channels // groups, 1, 1)
            # We want input channel i to map to output channel i.
            input_dim_per_group = self.in_channels // self.groups
            kernel_value = torch.zeros(self.out_channels, input_dim_per_group, 1, 1)

            for i in range(self.out_channels):
                # The input channel index within the group is i % input_dim_per_group
                kernel_value[i, i % input_dim_per_group, 0, 0] = 1

            dummy_conv.weight.data = kernel_value.to(self.identity.weight.device)

            # Fuse dummy conv with the identity BN
            fused_id = fuse_conv_bn(dummy_conv, self.identity)

            # Pad to 3x3
            w_id = F.pad(fused_id.weight, (1, 1, 1, 1))
            b_id = fused_id.bias

        # 4. Sum weights and biases
        final_w = w_3x3 + w_1x1 + w_id
        final_b = b_3x3 + b_1x1 + b_id

        # 5. Create the re-parameterized layer
        self.reparam_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = final_w
        self.reparam_conv.bias.data = final_b

        # Ensure device consistency
        self.reparam_conv.to(self.conv3x3.weight.device)

        # 6. Remove training branches to save memory
        del self.conv3x3, self.bn3x3, self.conv1x1, self.bn1x1, self.identity
        self.deploy = True


class UltraWideSERepNeXt(nn.Module):
    """
    Custom Ultra-Wide SE-RepNeXt with Spatial Feature Fusion.

    Architecture:
    - Stem: Standard Conv-BN-ReLU
    - Stage 1-3: Ultra-Wide RepNeXt Blocks
    - Head: Spatial Feature Fusion (Stage 2 + Stage 3) -> GAP -> Dense
    """

    def __init__(self, num_classes=1, deploy=False):
        super(UltraWideSERepNeXt, self).__init__()
        self.deploy = deploy

        # Configuration from Config
        channels = Config.BACKBONE_CHANNELS  # [96, 192, 384]
        groups = Config.CARDINALITY  # 32

        # --- Stem ---
        # Standard convolution to project RGB to initial width
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # --- Backbone ---
        # Stage 1: 96 -> 96 (Resolution: 32x32)
        self.stage1 = RepNeXtBlock(
            channels[0], channels[0], stride=1, groups=groups, deploy=deploy
        )

        # Stage 2: 96 -> 192 (Resolution: 16x16)
        self.stage2 = RepNeXtBlock(
            channels[0], channels[1], stride=2, groups=groups, deploy=deploy
        )

        # Stage 3: 192 -> 384 (Resolution: 8x8)
        self.stage3 = RepNeXtBlock(
            channels[1], channels[2], stride=2, groups=groups, deploy=deploy
        )

        # --- Spatial Feature Fusion Head ---
        # Projection: Stage 2 (192, 16x16) -> (192, 8x8)
        # Using a RepNeXt block for re-parameterizable projection
        self.fusion_proj = RepNeXtBlock(
            channels[1], channels[1], stride=2, groups=groups, deploy=deploy
        )

        # Mixing: Concat(192, 384) -> 576
        fusion_dim = channels[1] + channels[2]
        self.fusion_mix = nn.Sequential(
            nn.Conv2d(fusion_dim, fusion_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(fusion_dim),
            nn.ReLU(inplace=True),
        )

        # Classifier
        self.head = nn.Linear(fusion_dim, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Backbone
        x1 = self.stage1(x)  # 32x32
        x2 = self.stage2(x1)  # 16x16
        x3 = self.stage3(x2)  # 8x8

        # Spatial Fusion
        # Project Stage 2 feature map to Stage 3 resolution
        x2_proj = self.fusion_proj(x2)  # 16x16 -> 8x8

        # Concatenate along channel dimension
        x_fused = torch.cat([x2_proj, x3], dim=1)  # 192+384 = 576 channels

        # Mix features
        x_fused = self.fusion_mix(x_fused)

        # Global Average Pooling
        x_pool = F.adaptive_avg_pool2d(x_fused, 1).flatten(1)

        # Classification
        logits = self.head(x_pool)

        return logits

    def switch_to_deploy(self):
        """
        Converts the entire network to inference mode by fusing all RepNeXt blocks.
        """
        if self.deploy:
            return
        self.stage1.switch_to_deploy()
        self.stage2.switch_to_deploy()
        self.stage3.switch_to_deploy()
        self.fusion_proj.switch_to_deploy()
        self.deploy = True
