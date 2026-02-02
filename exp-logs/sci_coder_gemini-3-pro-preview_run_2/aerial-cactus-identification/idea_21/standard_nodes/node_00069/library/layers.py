import torch
import torch.nn as nn
import torch.nn.functional as F


class rSoftMax(nn.Module):
    """
    Radix-based Softmax for Split-Attention.
    Applies Softmax over the radix dimension to generate attention weights for each split.
    """

    def __init__(self, radix, cardinality):
        super(rSoftMax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            # x shape: [Batch, Cardinality, Radix, Width_per_group, 1, 1]
            # We apply softmax over the Radix dimension (dim=2)
            x = F.softmax(x, dim=2)
        else:
            # Fallback to Sigmoid for radix=1 (standard SE)
            x = torch.sigmoid(x)
        return x


class SplitAttention(nn.Module):
    """
    Split-Attention Module (Radix-2).
    Generalizes Squeeze-and-Excitation by splitting channels into radices and groups,
    and fusing them via adaptive attention.
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        radix=2,
        reduction_factor=4,
    ):
        super(SplitAttention, self).__init__()
        self.radix = radix
        self.cardinality = groups
        self.channels = channels

        # Calculate intermediate channels for the attention MLP
        # Ensure a minimum of 32 channels for stability
        inter_channels = max(channels * radix // reduction_factor, 32)

        # The main convolution:
        # Output channels = channels * radix
        # Groups = cardinality * radix
        # This performs the convolution for all splits simultaneously
        self.conv = nn.Conv2d(
            channels,
            channels * radix,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups * radix,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(channels * radix)
        self.relu0 = nn.ReLU(inplace=True)

        # Attention MLP (Squeeze-and-Excitation style)
        # Global Average Pooling is applied in forward()

        # FC1: Reduce
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # FC2: Expand to generate weights for each split
        self.fc2 = nn.Conv2d(
            inter_channels, channels * radix, 1, groups=self.cardinality
        )

        self.rsoftmax = rSoftMax(radix, groups)

    def forward(self, x):
        # 1. Convolution
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu0(x)

        batch, r_channels, h, w = x.size()

        if self.radix > 1:
            # 2. Reshape to expose Cardinality and Radix dimensions
            # Layout: [Batch, Cardinality, Radix, Width_per_group, H, W]
            width_per_group = self.channels // self.cardinality
            x = x.view(batch, self.cardinality, self.radix, width_per_group, h, w)

            # 3. Sum over Radix dimension to get U' (for Squeeze operation)
            x_gap = x.sum(dim=2)  # [B, C, W_pg, H, W]

            # 4. Global Average Pooling
            x_gap = x_gap.view(batch, self.channels, h, w)
            x_gap = F.adaptive_avg_pool2d(x_gap, 1)

            # 5. Attention MLP
            x_gap = self.fc1(x_gap)
            x_gap = self.bn1(x_gap)
            x_gap = self.relu1(x_gap)

            attn = self.fc2(x_gap)  # [B, channels*radix, 1, 1]

            # 6. Reshape attention scores
            attn = attn.view(batch, self.cardinality, self.radix, width_per_group, 1, 1)

            # 7. r-Softmax
            attn = self.rsoftmax(attn)

            # 8. Weighted Sum (Fusion)
            out = (x * attn).sum(dim=2)  # Sum over radix
            out = out.view(batch, self.channels, h, w)
        else:
            out = x

        return out


class ResNeStBlock(nn.Module):
    """
    ResNeSt Bottleneck Block.
    Uses Split-Attention as the central convolution.
    Strictly uses 1x1 convolutions for projection shortcuts.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        radix=2,
        cardinality=1,
        bottleneck_width_ratio=1.0,
    ):
        super(ResNeStBlock, self).__init__()

        # "Wide" configuration: Bottleneck width is derived from out_channels
        # bottleneck_width_ratio=1.0 means width equals out_channels (Expansion=1 relative to width)
        width = int(out_channels * bottleneck_width_ratio)

        # Ensure width is divisible by cardinality
        if width % cardinality != 0:
            width = ((width // cardinality) + 1) * cardinality

        self.in_channels = in_channels
        self.out_channels = out_channels

        # 1. 1x1 Conv (Reduce / Expand to width)
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu1 = nn.ReLU(inplace=True)

        # 2. Split-Attention Conv (3x3)
        self.sa = SplitAttention(
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            radix=radix,
        )
        self.bn_sa = nn.BatchNorm2d(width)
        self.relu_sa = nn.ReLU(inplace=True)

        # 3. 1x1 Conv (Restore / Expand to out_channels)
        self.conv2 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 4. Shortcut Connection
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            # Strictly 1x1 convolution for projection
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.sa(out)
        out = self.bn_sa(out)
        out = self.relu_sa(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        return out


class CactusNet(nn.Module):
    """
    Custom Wide Split-Attention Network.
    - Backbone: 3 Stages of ResNeSt Blocks.
    - Head: Multi-Scale Aggregation (Stage 2 + Stage 3).
    """

    def __init__(
        self,
        input_channels=3,
        num_classes=1,
        stages_channels=[64, 128, 256],
        radix=2,
        cardinality=1,
    ):
        super(CactusNet, self).__init__()

        # Stem: Initial processing
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stages_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stages_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32 -> 32x32 (Stride 1)
        # Input: stages_channels[0] -> Output: stages_channels[0]
        # We use a single heavy block per stage for this small image size
        self.layer1 = ResNeStBlock(
            stages_channels[0],
            stages_channels[0],
            stride=1,
            radix=radix,
            cardinality=cardinality,
        )

        # Stage 2: 32x32 -> 16x16 (Stride 2)
        # Input: stages_channels[0] -> Output: stages_channels[1]
        self.layer2 = ResNeStBlock(
            stages_channels[0],
            stages_channels[1],
            stride=2,
            radix=radix,
            cardinality=cardinality,
        )

        # Stage 3: 16x16 -> 8x8 (Stride 2)
        # Input: stages_channels[1] -> Output: stages_channels[2]
        self.layer3 = ResNeStBlock(
            stages_channels[1],
            stages_channels[2],
            stride=2,
            radix=radix,
            cardinality=cardinality,
        )

        # Classification Head
        # Concatenates GAP features from Stage 2 (16x16) and Stage 3 (8x8)
        self.fc = nn.Linear(stages_channels[1] + stages_channels[2], num_classes)

    def forward(self, x):
        # Backbone
        x = self.stem(x)  # 32x32
        x1 = self.layer1(x)  # 32x32
        x2 = self.layer2(x1)  # 16x16
        x3 = self.layer3(x2)  # 8x8

        # Multi-Scale Aggregation Head
        # GAP on Stage 2
        gap2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        # GAP on Stage 3
        gap3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)

        # Concatenate
        features = torch.cat([gap2, gap3], dim=1)

        # Classifier
        out = self.fc(features)

        return out
