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
    Deeply Supervised ResUNet++ (Nested U-Net).
    Features:
    - Nested dense skip pathways.
    - Residual Blocks with SE and SiLU.
    - Deep Supervision (multi-head output).
    """

    def __init__(self):
        super().__init__()

        in_ch = Config.IN_CHANNELS
        out_ch = Config.OUT_CHANNELS
        base_ch = Config.BASE_CHANNELS
        self.deep_supervision = Config.DEEP_SUPERVISION

        # Filter counts: [32, 64, 128, 256, 512]
        filters = [base_ch * (2**i) for i in range(5)]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # --- Encoder (Backbone) ---
        self.conv0_0 = ResBlock(in_ch, filters[0])
        self.conv1_0 = ResBlock(filters[0], filters[1])
        self.conv2_0 = ResBlock(filters[1], filters[2])
        self.conv3_0 = ResBlock(filters[2], filters[3])
        self.conv4_0 = ResBlock(filters[3], filters[4])

        # --- Nested Skip Pathways ---
        # Level 0 (Output Resolution)
        # Inputs: [0_0, Up(1_0)]
        self.conv0_1 = ResBlock(filters[0] + filters[1], filters[0])
        # Inputs: [0_0, 0_1, Up(1_1)]
        self.conv0_2 = ResBlock(filters[0] * 2 + filters[1], filters[0])
        # Inputs: [0_0, 0_1, 0_2, Up(1_2)]
        self.conv0_3 = ResBlock(filters[0] * 3 + filters[1], filters[0])
        # Inputs: [0_0, 0_1, 0_2, 0_3, Up(1_3)]
        self.conv0_4 = ResBlock(filters[0] * 4 + filters[1], filters[0])

        # Level 1
        # Inputs: [1_0, Up(2_0)]
        self.conv1_1 = ResBlock(filters[1] + filters[2], filters[1])
        # Inputs: [1_0, 1_1, Up(2_1)]
        self.conv1_2 = ResBlock(filters[1] * 2 + filters[2], filters[1])
        # Inputs: [1_0, 1_1, 1_2, Up(2_2)]
        self.conv1_3 = ResBlock(filters[1] * 3 + filters[2], filters[1])

        # Level 2
        # Inputs: [2_0, Up(3_0)]
        self.conv2_1 = ResBlock(filters[2] + filters[3], filters[2])
        # Inputs: [2_0, 2_1, Up(3_1)]
        self.conv2_2 = ResBlock(filters[2] * 2 + filters[3], filters[2])

        # Level 3
        # Inputs: [3_0, Up(4_0)]
        self.conv3_1 = ResBlock(filters[3] + filters[4], filters[3])

        # --- Output Heads (Deep Supervision) ---
        self.final0_1 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final0_2 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final0_3 = nn.Conv2d(filters[0], out_ch, kernel_size=1)
        self.final0_4 = nn.Conv2d(filters[0], out_ch, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # --- Decoder / Nested Pathways ---

        # Level 3
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        # Level 2
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        # Level 1
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        # Level 0
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # --- Output ---
        out4 = self.final0_4(x0_4)

        # Deep Supervision: Return list of outputs from all Level 0 nodes
        if self.deep_supervision and self.training:
            out1 = self.final0_1(x0_1)
            out2 = self.final0_2(x0_2)
            out3 = self.final0_3(x0_3)
            return [out1, out2, out3, out4]

        # Inference: Return only the final (deepest) output
        return out4
