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


class ResUNetPlusPlus(nn.Module):
    """
    Standard ResUNet Architecture.
    Simplified from ResUNet++ to allow for larger patch sizes and faster convergence.
    Features:
    - Standard U-Net Encoder-Decoder topology.
    - Residual Blocks with SE and SiLU.
    - Transposed Convolution for upsampling (Cite Lesson 00020).
    """

    def __init__(self):
        super(ResUNetPlusPlus, self).__init__()

        # Filter configuration: [64, 128, 256, 512, 1024]
        nb_filter = [Config.BASE_FILTERS * (2**i) for i in range(5)]

        # Downsampling
        self.pool = nn.MaxPool2d(2, 2)

        # Helper for Transposed Convolution Upsampling
        def up_layer(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # ---------------------------------------------------------------------
        # Encoder (Backbone)
        # ---------------------------------------------------------------------
        self.conv0 = ResidualBlock(1, nb_filter[0])
        self.conv1 = ResidualBlock(nb_filter[0], nb_filter[1])
        self.conv2 = ResidualBlock(nb_filter[1], nb_filter[2])
        self.conv3 = ResidualBlock(nb_filter[2], nb_filter[3])
        self.conv4 = ResidualBlock(nb_filter[3], nb_filter[4])

        # ---------------------------------------------------------------------
        # Decoder Path
        # ---------------------------------------------------------------------

        # Level 3
        self.up4 = up_layer(nb_filter[4], nb_filter[3])
        self.conv3_dec = ResidualBlock(nb_filter[3] + nb_filter[3], nb_filter[3])

        # Level 2
        self.up3 = up_layer(nb_filter[3], nb_filter[2])
        self.conv2_dec = ResidualBlock(nb_filter[2] + nb_filter[2], nb_filter[2])

        # Level 1
        self.up2 = up_layer(nb_filter[2], nb_filter[1])
        self.conv1_dec = ResidualBlock(nb_filter[1] + nb_filter[1], nb_filter[1])

        # Level 0
        self.up1 = up_layer(nb_filter[1], nb_filter[0])
        self.conv0_dec = ResidualBlock(nb_filter[0] + nb_filter[0], nb_filter[0])

        # Final Output
        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv0(x)
        x1 = self.conv1(self.pool(x0))
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))

        # Decoder
        u4 = self.up4(x4)

        diffY = x3.size()[2] - u4.size()[2]
        diffX = x3.size()[3] - u4.size()[3]
        u4 = F.pad(u4, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        d4 = self.conv3_dec(torch.cat([x3, u4], dim=1))

        u3 = self.up3(d4)

        diffY = x2.size()[2] - u3.size()[2]
        diffX = x2.size()[3] - u3.size()[3]
        u3 = F.pad(u3, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        d3 = self.conv2_dec(torch.cat([x2, u3], dim=1))

        u2 = self.up2(d3)

        diffY = x1.size()[2] - u2.size()[2]
        diffX = x1.size()[3] - u2.size()[3]
        u2 = F.pad(u2, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        d2 = self.conv1_dec(torch.cat([x1, u2], dim=1))

        u1 = self.up1(d2)

        diffY = x0.size()[2] - u1.size()[2]
        diffX = x0.size()[3] - u1.size()[3]
        u1 = F.pad(u1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        d1 = self.conv0_dec(torch.cat([x0, u1], dim=1))

        out = self.final(d1)
        return out
