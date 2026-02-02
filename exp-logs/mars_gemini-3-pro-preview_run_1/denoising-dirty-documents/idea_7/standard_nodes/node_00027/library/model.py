import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling followed by convolution as per requirements
        # to avoid checkerboard artifacts associated with Transposed Conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # The input to DoubleConv will be the concatenation of upsampled input and skip connection
            # in_channels here refers to the sum of channels from both inputs
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            # Fallback (not used in this config)
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Input from the lower layer (to be upsampled)
        x2: Skip connection from the encoder (to be concatenated)
        """
        x1 = self.up(x1)

        # Handle potential padding issues if dimensions are not perfect multiples
        # (Though pad_to_multiple in utils.py should handle this globally)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map features to output classes"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Resolution-Scaled 4-Level U-Net Architecture.
    """

    def __init__(self, n_channels=1, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Retrieve filter configuration from Config
        # Expected: [32, 64, 128, 256, 512]
        filters = Config.ENCODER_FILTERS

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, filters[0])  # 1 -> 32
        self.down1 = Down(filters[0], filters[1])  # 32 -> 64
        self.down2 = Down(filters[1], filters[2])  # 64 -> 128
        self.down3 = Down(filters[2], filters[3])  # 128 -> 256
        self.down4 = Down(filters[3], filters[4])  # 256 -> 512 (Bottleneck)

        # --- Decoder ---
        # Up blocks take (channels_from_below + skip_channels, out_channels)

        # Up1: Input from down4 (512) + Skip from down3 (256) -> Output 256
        self.up1 = Up(filters[4] + filters[3], filters[3])

        # Up2: Input from up1 (256) + Skip from down2 (128) -> Output 128
        self.up2 = Up(filters[3] + filters[2], filters[2])

        # Up3: Input from up2 (128) + Skip from down1 (64) -> Output 64
        self.up3 = Up(filters[2] + filters[1], filters[1])

        # Up4: Input from up3 (64) + Skip from inc (32) -> Output 32
        self.up4 = Up(filters[1] + filters[0], filters[0])

        # --- Output ---
        self.outc = OutConv(filters[0], n_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)

        # Apply Sigmoid to ensure output pixel intensities are in [0, 1]
        return torch.sigmoid(logits)
