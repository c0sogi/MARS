import torch
import torch.nn as nn
from library.config import Config


class ResidualBlock(nn.Module):
    """
    A standard Residual Block with support for dilated convolutions.
    Maintains spatial resolution by setting padding equal to dilation.
    """

    def __init__(self, channels, dilation):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class FFDCNet(nn.Module):
    """
    Feature-Fused Dilated Convolutional Network (FFDC-Net).

    Architecture:
    1. Volumetric Projection Stem (1x1 Conv + InstanceNorm)
    2. Texture Preservation Block (Low dilation ResBlocks)
    3. Context Aggregation Block (High dilation ResBlocks)
    4. Multi-Scale Fusion Head (Concatenation + 1x1 Conv)
    """

    def __init__(
        self, in_channels=Config.IN_CHANNELS, stem_channels=Config.STEM_CHANNELS
    ):
        super(FFDCNet, self).__init__()

        # 1. Volumetric Projection Stem
        # Projects 65 depth slices to dense feature map.
        # Uses InstanceNorm to decouple ink signal from global X-ray density.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(stem_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        # 2. Texture Preservation Block
        # Focuses on high-frequency details/edges.
        self.texture_block = nn.Sequential(
            ResidualBlock(stem_channels, dilation=1),
            ResidualBlock(stem_channels, dilation=1),
        )

        # 3. Context Aggregation Block
        # Expands receptive field to capture stroke continuity.
        # Dilations: 2, 4, 8, 16
        self.context_block = nn.Sequential(
            ResidualBlock(stem_channels, dilation=2),
            ResidualBlock(stem_channels, dilation=4),
            ResidualBlock(stem_channels, dilation=8),
            ResidualBlock(stem_channels, dilation=16),
        )

        # 4. Multi-Scale Fusion Head
        # Concatenates Texture (local) and Context (global) features.
        # Input channels = 64 (Texture) + 64 (Context) = 128
        self.classifier = nn.Conv2d(stem_channels * 2, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, 65, H, W)

        # Project volume to features
        x = self.stem(x)

        # Extract texture features
        feat_texture = self.texture_block(x)

        # Extract context features (fed from texture features)
        feat_context = self.context_block(feat_texture)

        # Fusion: Concatenate texture and context features
        # Shape: (Batch, 128, H, W)
        fused = torch.cat([feat_texture, feat_context], dim=1)

        # Final Classification
        # Output logits for numerical stability with BCEWithLogitsLoss
        logits = self.classifier(fused)

        return logits
