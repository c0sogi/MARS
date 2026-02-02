import torch
import torch.nn as nn
from library import config


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Block with Global Average Pooling.
    Acts as a low-pass filter for channel attention.
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv -> BN -> LeakyReLU -> SE -> MaxPool.
    Retains bias in Conv2d and uses LeakyReLU to preserve signal dynamics.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        # Bias is explicitly retained to preserve initialization dynamics
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(negative_slope=config.LEAKY_RELU_SLOPE, inplace=True)
        self.se = SELayer(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class SHH_SE_CNN(nn.Module):
    """
    Selective Hierarchical Hybrid-SE CNN.
    Features:
    - Plain CNN Backbone (4 Stages)
    - Hybrid Attention (SE with AvgPool)
    - Selective Hierarchical Aggregation (Stage 3 + Stage 4)
    - Raw Scale Fusion (Features + Incidence Angle)
    """

    def __init__(self):
        super(SHH_SE_CNN, self).__init__()

        # Channel Configuration: 64 -> 128 -> 128 -> 128
        c = config.CHANNEL_CONFIG

        # Backbone Stages
        self.block1 = ConvBlock(config.INPUT_CHANNELS, c[0])
        self.block2 = ConvBlock(c[0], c[1])
        self.block3 = ConvBlock(c[1], c[2])
        self.block4 = ConvBlock(c[2], c[3])

        # Readout Pooling: Global Max Pooling to capture peak signals
        self.global_max = nn.AdaptiveMaxPool2d(1)

        # Classification Head
        # Input: Stage 3 (128) + Stage 4 (128) + Angle (1) = 257
        # We use a hidden dimension of 256 for the single hidden layer.
        hidden_dim = 256

        self.head = nn.Sequential(
            nn.Linear(config.FUSION_INPUT_DIM, hidden_dim),
            nn.LeakyReLU(negative_slope=config.LEAKY_RELU_SLOPE, inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(hidden_dim, config.NUM_CLASSES),
        )

        # Initialization: Relying on PyTorch Default (Kaiming Uniform) as requested.

    def forward(self, x, angle):
        # Backbone Forward Pass
        x = self.block1(x)
        x = self.block2(x)

        # Stage 3
        x3 = self.block3(x)

        # Stage 4
        x4 = self.block4(x3)

        # Selective Hierarchical Readout
        # Apply Global Max Pooling to Stage 3 and Stage 4 feature maps
        p3 = self.global_max(x3).view(x3.size(0), -1)
        p4 = self.global_max(x4).view(x4.size(0), -1)

        # Angle Processing
        # Reshape angle to (Batch, 1)
        angle = angle.view(-1, 1)

        # Feature Fusion
        # Concatenate: [Stage 3 Features, Stage 4 Features, Raw Angle]
        fused = torch.cat([p3, p4, angle], dim=1)

        # Classification
        logits = self.head(fused)

        # Return flattened logits (B,)
        return logits.squeeze(1)
