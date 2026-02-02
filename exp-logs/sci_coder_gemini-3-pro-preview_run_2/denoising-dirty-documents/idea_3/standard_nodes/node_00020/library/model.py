import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses adaptively.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResBlock(nn.Module):
    """
    Residual Block with SiLU activation and Squeeze-and-Excitation.
    Structure: Input -> [Conv-BN-SiLU-Conv-BN-SE] + Input -> SiLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)

        self.act2 = nn.SiLU(inplace=True)

        # Shortcut connection to handle channel/stride changes
        self.shortcut = nn.Identity()
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
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = self.act2(out)
        return out


class ResUNet(nn.Module):
    """
    Standard ResUNet.
    Features:
    - Residual Blocks with SE and SiLU.
    - Standard U-Net connections (no nested paths).
    """

    def __init__(self):
        super().__init__()

        in_ch = Config.IN_CHANNELS
        out_ch = Config.OUT_CHANNELS
        base_ch = Config.BASE_CHANNELS

        # Filter counts: [32, 64, 128, 256, 512]
        filters = [base_ch * (2**i) for i in range(5)]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # --- Encoder (Backbone) ---
        self.conv0 = ResBlock(in_ch, filters[0])
        self.conv1 = ResBlock(filters[0], filters[1])
        self.conv2 = ResBlock(filters[1], filters[2])
        self.conv3 = ResBlock(filters[2], filters[3])
        self.conv4 = ResBlock(filters[3], filters[4])

        # --- Decoder ---
        # Up3: Input = Enc3 + Up(Enc4)
        self.conv_up3 = ResBlock(filters[3] + filters[4], filters[3])

        # Up2: Input = Enc2 + Up(Dec3)
        self.conv_up2 = ResBlock(filters[2] + filters[3], filters[2])

        # Up1: Input = Enc1 + Up(Dec2)
        self.conv_up1 = ResBlock(filters[1] + filters[2], filters[1])

        # Up0: Input = Enc0 + Up(Dec1)
        self.conv_up0 = ResBlock(filters[0] + filters[1], filters[0])

        # --- Output Head ---
        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.conv0(x)
        x1 = self.conv1(self.pool(x0))
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))

        # --- Decoder ---
        x3_up = self.conv_up3(torch.cat([x3, self.up(x4)], 1))
        x2_up = self.conv_up2(torch.cat([x2, self.up(x3_up)], 1))
        x1_up = self.conv_up1(torch.cat([x1, self.up(x2_up)], 1))
        x0_up = self.conv_up0(torch.cat([x0, self.up(x1_up)], 1))

        # --- Output ---
        out = self.final(x0_up)

        return out
