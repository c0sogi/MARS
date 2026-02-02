import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bilinear upsampling followed by a convolution to reduce channels
        # to match the skip connection size
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: input from lower level (to be upsampled)
        # x2: skip connection from encoder
        x1 = self.up(x1)
        x1 = self.conv(x1)

        # Handle padding if dimensions don't match exactly due to odd sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class DilatedBottleneck(nn.Module):
    """
    Central bottleneck with dilated convolutions to expand receptive field
    without reducing resolution or adding parameters.
    """

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # Dilation 2
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=2, dilation=2, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # Dilation 4
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=4, dilation=4, bias=False
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualShallowUNet(nn.Module):
    """
    Residual Shallow U-Net with Dilated Bottleneck.
    Predicts the noise residual (Input - Target).
    """

    def __init__(self, n_channels=1, n_classes=1):
        super(ResidualShallowUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder: 32 -> 64 -> 128
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)

        # Bottleneck: Dilated Convolutions
        self.bottleneck = DilatedBottleneck(128)

        # Decoder: 128 -> 64 -> 32
        self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)

        # Output projection
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # -> 32
        x2 = self.down1(x1)  # -> 64
        x3 = self.down2(x2)  # -> 128

        # Bottleneck
        x_bot = self.bottleneck(x3)  # -> 128

        # Decoder
        x = self.up1(x_bot, x2)  # -> 64
        x = self.up2(x, x1)  # -> 32

        # Output (Predicted Noise)
        logits = self.outc(x)
        return logits
