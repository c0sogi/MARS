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


class ResUNetPlusPlus(nn.Module):
    """
    Standard ResUNet Architecture (Simplified from PlusPlus).
    Replaces ResUNet++ to avoid convergence tax (Cite solution_lesson_node_00019).
    Uses Transposed Convolutions for learnable upsampling (Cite solution_lesson_node_00020).
    Retains ResBlocks with SE and SiLU (Cite solution_lesson_node_00016).
    """

    def __init__(self):
        super().__init__()

        in_ch = Config.IN_CHANNELS
        out_ch = Config.OUT_CHANNELS
        base_ch = Config.BASE_CHANNELS

        # Filter counts: [64, 128, 256, 512, 1024] (if base=64)
        self.filters = [base_ch * (2**i) for i in range(5)]

        self.pool = nn.MaxPool2d(2, 2)

        # --- Encoder ---
        self.enc1 = ResBlock(in_ch, self.filters[0])
        self.enc2 = ResBlock(self.filters[0], self.filters[1])
        self.enc3 = ResBlock(self.filters[1], self.filters[2])
        self.enc4 = ResBlock(self.filters[2], self.filters[3])

        # --- Bridge ---
        self.bridge = ResBlock(self.filters[3], self.filters[4])

        # --- Decoder ---
        # Up 4->3
        self.up4 = nn.ConvTranspose2d(
            self.filters[4], self.filters[3], kernel_size=2, stride=2
        )
        self.dec4 = ResBlock(self.filters[3] * 2, self.filters[3])

        # Up 3->2
        self.up3 = nn.ConvTranspose2d(
            self.filters[3], self.filters[2], kernel_size=2, stride=2
        )
        self.dec3 = ResBlock(self.filters[2] * 2, self.filters[2])

        # Up 2->1
        self.up2 = nn.ConvTranspose2d(
            self.filters[2], self.filters[1], kernel_size=2, stride=2
        )
        self.dec2 = ResBlock(self.filters[1] * 2, self.filters[1])

        # Up 1->0
        self.up1 = nn.ConvTranspose2d(
            self.filters[1], self.filters[0], kernel_size=2, stride=2
        )
        self.dec1 = ResBlock(self.filters[0] * 2, self.filters[0])

        # --- Output ---
        self.final = nn.Conv2d(self.filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bridge
        b = self.bridge(self.pool(e4))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.final(d1)
