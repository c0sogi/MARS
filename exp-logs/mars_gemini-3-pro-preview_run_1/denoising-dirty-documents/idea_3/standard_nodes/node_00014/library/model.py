import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """
    Upscaling => Concatenation => DoubleConv
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Bilinear upsampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution after concatenation
        # Input channels = in_channels (upsampled) + skip_channels
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        # x1: Feature from the previous decoder layer (to be upsampled)
        # x2: Feature from the encoder skip connection

        x1 = self.up(x1)

        # Handle padding if shapes don't match exactly (e.g. odd input dimensions)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

        # Concatenate
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture for image denoising.
    """

    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        # Level 1: 1 -> 32
        self.inc = DoubleConv(n_channels, 32)

        # Level 2: 32 -> 64
        self.down1 = DoubleConv(32, 64)

        # Level 3: 64 -> 128
        self.down2 = DoubleConv(64, 128)

        # Level 4: 128 -> 256
        self.down3 = DoubleConv(128, 256)

        # Level 5 (Bottleneck): 256 -> 512
        self.down4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        # Up 1: 512 + 256 -> 256
        self.up1 = Up(512, 256, 256)

        # Up 2: 256 + 128 -> 128
        self.up2 = Up(256, 128, 128)

        # Up 3: 128 + 64 -> 64
        self.up3 = Up(128, 64, 64)

        # Up 4: 64 + 32 -> 32
        self.up4 = Up(64, 32, 32)

        # Output mapping
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)  # 32 channels

        x2 = self.pool(x1)
        x2 = self.down1(x2)  # 64 channels

        x3 = self.pool(x2)
        x3 = self.down2(x3)  # 128 channels

        x4 = self.pool(x3)
        x4 = self.down3(x4)  # 256 channels

        x5 = self.pool(x4)
        x5 = self.down4(x5)  # 512 channels (Bottleneck)

        # Decoder Path
        x = self.up1(x5, x4)  # -> 256
        x = self.up2(x, x3)  # -> 128
        x = self.up3(x, x2)  # -> 64
        x = self.up4(x, x1)  # -> 32

        logits = self.outc(x)
        return logits
