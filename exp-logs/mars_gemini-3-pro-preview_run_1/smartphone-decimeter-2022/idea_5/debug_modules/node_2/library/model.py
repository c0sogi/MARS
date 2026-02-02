import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with optional dilation and dropout.
    Structure: Conv1 -> BN -> ReLU -> Dropout -> Conv2 -> BN -> (+ Shortcut) -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        dropout=0.0,
    ):
        super(ResidualBlock1D, self).__init__()

        # Calculate padding to maintain sequence length for stride=1
        # For stride > 1, this padding logic is valid for 'same' like behavior with odd kernels
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResUNet1D(nn.Module):
    """
    1D Residual U-Net for GNSS sequence error correction.
    Predicts (DeltaEast, DeltaNorth) residuals in meters.
    """

    def __init__(self, config=Config):
        super(ResUNet1D, self).__init__()

        self.in_channels = config.IN_CHANNELS
        self.out_channels = config.OUT_CHANNELS
        self.base_channels = config.MODEL_CHANNELS
        self.dropout = config.DROPOUT

        # --- Encoder ---
        # Layer 1
        self.enc1 = ResidualBlock1D(
            self.in_channels, self.base_channels, dropout=self.dropout
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 2
        self.enc2 = ResidualBlock1D(
            self.base_channels, self.base_channels * 2, dropout=self.dropout
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 3
        self.enc3 = ResidualBlock1D(
            self.base_channels * 2, self.base_channels * 4, dropout=self.dropout
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Layer 4
        self.enc4 = ResidualBlock1D(
            self.base_channels * 4, self.base_channels * 8, dropout=self.dropout
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        # --- Bottleneck (Dilated Convolutions) ---
        self.bottleneck1 = ResidualBlock1D(
            self.base_channels * 8,
            self.base_channels * 8,
            dilation=2,
            dropout=self.dropout,
        )
        self.bottleneck2 = ResidualBlock1D(
            self.base_channels * 8,
            self.base_channels * 8,
            dilation=4,
            dropout=self.dropout,
        )

        # --- Decoder ---
        # Layer 4
        self.up4 = nn.ConvTranspose1d(
            self.base_channels * 8, self.base_channels * 4, kernel_size=2, stride=2
        )
        self.dec4 = ResidualBlock1D(
            self.base_channels * 8 + self.base_channels * 4,  # Concat: Enc4 + Up
            self.base_channels * 4,
            dropout=self.dropout,
        )

        # Layer 3
        self.up3 = nn.ConvTranspose1d(
            self.base_channels * 4, self.base_channels * 2, kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock1D(
            self.base_channels * 4 + self.base_channels * 2,  # Concat: Enc3 + Up
            self.base_channels * 2,
            dropout=self.dropout,
        )

        # Layer 2
        self.up2 = nn.ConvTranspose1d(
            self.base_channels * 2, self.base_channels, kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock1D(
            self.base_channels * 2 + self.base_channels,  # Concat: Enc2 + Up
            self.base_channels,
            dropout=self.dropout,
        )

        # Layer 1
        self.up1 = nn.ConvTranspose1d(
            self.base_channels, self.base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock1D(
            self.base_channels + self.base_channels,  # Concat: Enc1 + Up
            self.base_channels,
            dropout=self.dropout,
        )

        # --- Head ---
        self.head = nn.Conv1d(self.base_channels, self.out_channels, kernel_size=1)

    def forward(self, x):
        # x: (Batch, In_Channels, Length)

        # Encoder
        e1 = self.enc1(x)  # (B, 64, L)
        p1 = self.pool1(e1)  # (B, 64, L/2)

        e2 = self.enc2(p1)  # (B, 128, L/2)
        p2 = self.pool2(e2)  # (B, 128, L/4)

        e3 = self.enc3(p2)  # (B, 256, L/4)
        p3 = self.pool3(e3)  # (B, 256, L/8)

        e4 = self.enc4(p3)  # (B, 512, L/8)
        p4 = self.pool4(e4)  # (B, 512, L/16)

        # Bottleneck
        b = self.bottleneck1(p4)
        b = self.bottleneck2(b)  # (B, 512, L/16)

        # Decoder
        d4 = self.up4(b)  # (B, 256, L/8)
        # Handle potential padding issues if length was odd
        if d4.size(2) != e4.size(2):
            d4 = F.interpolate(d4, size=e4.size(2), mode="linear", align_corners=False)
        d4 = torch.cat([e4, d4], dim=1)  # (B, 512+256, L/8)
        d4 = self.dec4(d4)  # (B, 256, L/8)

        d3 = self.up3(d4)  # (B, 128, L/4)
        if d3.size(2) != e3.size(2):
            d3 = F.interpolate(d3, size=e3.size(2), mode="linear", align_corners=False)
        d3 = torch.cat([e3, d3], dim=1)  # (B, 256+128, L/4)
        d3 = self.dec3(d3)  # (B, 128, L/4)

        d2 = self.up2(d3)  # (B, 64, L/2)
        if d2.size(2) != e2.size(2):
            d2 = F.interpolate(d2, size=e2.size(2), mode="linear", align_corners=False)
        d2 = torch.cat([e2, d2], dim=1)  # (B, 128+64, L/2)
        d2 = self.dec2(d2)  # (B, 64, L/2)

        d1 = self.up1(d2)  # (B, 64, L)
        if d1.size(2) != e1.size(2):
            d1 = F.interpolate(d1, size=e1.size(2), mode="linear", align_corners=False)
        d1 = torch.cat([e1, d1], dim=1)  # (B, 64+64, L)
        d1 = self.dec1(d1)  # (B, 64, L)

        # Head
        out = self.head(d1)  # (B, 2, L)

        return out
