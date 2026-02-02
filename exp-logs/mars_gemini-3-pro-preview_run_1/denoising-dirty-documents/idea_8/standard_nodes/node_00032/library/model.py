import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import MODEL_CHANNELS


class DoubleConv(nn.Module):
    """
    (Conv2d => BN => ReLU) * 2
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


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the lower layer (to be upsampled)
        # x2 is the skip connection from the encoder
        x1 = self.up(x1)

        # Handle potential padding issues if dimensions are not perfect multiples
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    4-Level Standard U-Net
    """

    def __init__(self):
        super().__init__()

        # Configuration
        c1, c2, c3, c4 = MODEL_CHANNELS

        # Encoder
        self.inc = DoubleConv(1, c1)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c1, c2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c2, c3))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c3, c4))

        # Bottleneck
        self.bot = nn.Sequential(nn.MaxPool2d(2), DoubleConv(c4, c4 * 2))

        # Decoder
        # Up1 inputs: bottleneck output (c4*2) + skip connection from down3 (c4)
        self.up1 = Up(c4 * 2 + c4, c4)
        self.up2 = Up(c4 + c3, c3)
        self.up3 = Up(c3 + c2, c2)
        self.up4 = Up(c2 + c1, c1)

        # Output mapping
        self.outc = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x_bot = self.bot(x4)

        # Decoder
        x = self.up1(x_bot, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output
        logits = self.outc(x)
        return torch.sigmoid(logits)
