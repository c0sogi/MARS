import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DoubleConv(nn.Module):
    """
    (Conv => BN => ReLU) * 2
    Uses reflection padding to maintain boundary continuity for noise statistics.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                padding_mode=Config.PADDING_MODE,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=Config.PADDING_MODE,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Bilinear Upsampling followed by Convolution
        # We use bilinear upsampling to avoid checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 1x1 Convolution to reduce channels after upsampling, mimicking TransposeConv behavior
        # Maps in_channels (e.g., 512) to in_channels // 2 (e.g., 256)
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)

        # DoubleConv takes the concatenated input (upsampled + skip connection)
        # Input channels = (in_channels // 2) + (in_channels // 2) = in_channels
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: input from previous layer (bottom)
        # x2: skip connection from encoder

        x1 = self.up(x1)
        x1 = self.conv(x1)

        # Handle padding issues if dimensions don't match exactly
        # (e.g. if input size was not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1,
                [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
                mode=Config.PADDING_MODE,
            )

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard 4-Level U-Net with Signal-Aligned configuration.
    Encoder: 32 -> 64 -> 128 -> 256 -> 512
    Decoder: 512 -> 256 -> 128 -> 64 -> 32
    """

    def __init__(self, n_channels=Config.IN_CHANNELS, n_classes=Config.OUT_CHANNELS):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Filter counts based on 4-Level depth
        f1 = 32
        f2 = 64
        f3 = 128
        f4 = 256
        f5 = 512

        self.inc = DoubleConv(n_channels, f1)
        self.down1 = Down(f1, f2)
        self.down2 = Down(f2, f3)
        self.down3 = Down(f3, f4)
        self.down4 = Down(f4, f5)

        self.up1 = Up(f5, f4)
        self.up2 = Up(f4, f3)
        self.up3 = Up(f3, f2)
        self.up4 = Up(f2, f1)
        self.outc = OutConv(f1, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
