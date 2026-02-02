import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DoubleConv(nn.Module):
    """
    (Conv2d => BN => ReLU) * 2
    Standard building block for U-Net.
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


class Down(nn.Module):
    """
    Downscaling with MaxPool then DoubleConv.
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
    Upscaling then DoubleConv.
    Uses Bilinear interpolation for upsampling followed by concatenation with skip connection.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Using bilinear upsampling as it is parameter-free and effective
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After concat, channels = in_channels (from up) + skip_channels
            # We assume in_channels passed to __init__ accounts for the concatenated size
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            # Alternative: Transposed Convolution
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Input from previous decoder layer
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)

        # Input is CHW. Handle potential rounding errors in padding if sizes don't match exactly
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX > 0 or diffY > 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TemporalAshNet(nn.Module):
    """
    Custom U-Net architecture for Contrail Detection using Temporal Ash Composites.

    Input: (B, 9, 256, 256) - 3 timesteps x 3 Ash channels
    Output: (B, 1, 256, 256) - Logits for binary mask

    Structure:
    - 4 Encoder Stages (max downsampling 16x)
    - Channels: [64, 128, 256, 512]
    - Bottleneck at 16x16 resolution
    - Symmetric Decoder with Skip Connections
    """

    def __init__(self):
        super(TemporalAshNet, self).__init__()

        n_channels = Config.IN_CHANNELS
        n_classes = Config.NUM_CLASSES

        # --- Encoder ---
        # Initial Block: 9 -> 64
        self.inc = DoubleConv(n_channels, 64)

        # Down 1: 64 -> 128 (128x128)
        self.down1 = Down(64, 128)

        # Down 2: 128 -> 256 (64x64)
        self.down2 = Down(128, 256)

        # Down 3: 256 -> 512 (32x32)
        self.down3 = Down(256, 512)

        # Down 4 (Bottleneck): 512 -> 512 (16x16)
        # We keep channels at 512 to remain lightweight and avoid excessive parameters
        self.down4 = Down(512, 512)

        # --- Decoder ---
        # Up 1: Input (512 from bottleneck + 512 from down3) = 1024 -> Output 256
        self.up1 = Up(1024, 256)

        # Up 2: Input (256 from up1 + 256 from down2) = 512 -> Output 128
        self.up2 = Up(512, 128)

        # Up 3: Input (128 from up2 + 128 from down1) = 256 -> Output 64
        self.up3 = Up(256, 64)

        # Up 4: Input (64 from up3 + 64 from inc) = 128 -> Output 64
        self.up4 = Up(128, 64)

        # --- Output Head ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)  # (B, 64, 256, 256)
        x2 = self.down1(x1)  # (B, 128, 128, 128)
        x3 = self.down2(x2)  # (B, 256, 64, 64)
        x4 = self.down3(x3)  # (B, 512, 32, 32)
        x5 = self.down4(x4)  # (B, 512, 16, 16) - Bottleneck

        # Decoder Path
        x = self.up1(x5, x4)  # (B, 256, 32, 32)
        x = self.up2(x, x3)  # (B, 128, 64, 64)
        x = self.up3(x, x2)  # (B, 64, 128, 128)
        x = self.up4(x, x1)  # (B, 64, 256, 256)

        # Output Logits
        logits = self.outc(x)  # (B, 1, 256, 256)
        return logits
