import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.
    Recalibrates feature maps by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        # Ensure reduction doesn't make hidden dim 0
        hidden_dim = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Residual Block with SiLU activation and Squeeze-and-Excitation.
    Standard building block for ResUNet.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.use_se = Config.USE_SE

        # Activation function configuration
        self.activation = (
            nn.SiLU(inplace=True) if Config.USE_SILU else nn.ReLU(inplace=True)
        )

        # First Convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second Convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Squeeze and Excitation
        if self.use_se:
            self.se = SqueezeExcitation(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
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
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out += residual
        out = self.activation(out)
        return out


class ResUNet(nn.Module):
    """
    Standard ResUNet Architecture.
    Simplified from ResUNet++ to allow for larger patch sizes and better context.
    Cite solution_lesson_node_00022.
    """

    def __init__(self):
        super(ResUNet, self).__init__()

        # Filter configuration: [64, 128, 256, 512, 1024]
        nb_filter = [Config.BASE_FILTERS * (2**i) for i in range(5)]

        # Downsampling
        self.pool = nn.MaxPool2d(2, 2)

        # Helper for Transposed Convolution Upsampling
        # Cite solution_lesson_node_00020 (Prefer Learnable Upsampling)
        def up_layer(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        self.conv0 = ResidualBlock(1, nb_filter[0])
        self.conv1 = ResidualBlock(nb_filter[0], nb_filter[1])
        self.conv2 = ResidualBlock(nb_filter[1], nb_filter[2])
        self.conv3 = ResidualBlock(nb_filter[2], nb_filter[3])
        self.conv4 = ResidualBlock(nb_filter[3], nb_filter[4])

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        # Level 3
        self.up4 = up_layer(nb_filter[4], nb_filter[3])
        self.conv4_dec = ResidualBlock(nb_filter[4], nb_filter[3])  # 512+512 -> 512

        # Level 2
        self.up3 = up_layer(nb_filter[3], nb_filter[2])
        self.conv3_dec = ResidualBlock(nb_filter[3], nb_filter[2])  # 256+256 -> 256

        # Level 1
        self.up2 = up_layer(nb_filter[2], nb_filter[1])
        self.conv2_dec = ResidualBlock(nb_filter[2], nb_filter[1])  # 128+128 -> 128

        # Level 0
        self.up1 = up_layer(nb_filter[1], nb_filter[0])
        self.conv1_dec = ResidualBlock(nb_filter[1], nb_filter[0])  # 64+64 -> 64

        # Final
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv0(x)
        x1 = self.conv1(self.pool(x0))
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))

        # Decoder
        d4 = self.up4(x4)
        d4 = torch.cat([x3, d4], dim=1)
        d4 = self.conv4_dec(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.conv3_dec(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([x1, d2], dim=1)
        d2 = self.conv2_dec(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([x0, d1], dim=1)
        d1 = self.conv1_dec(d1)

        return self.final(d1)
