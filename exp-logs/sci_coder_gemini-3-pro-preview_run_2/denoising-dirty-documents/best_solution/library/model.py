import torch
import torch.nn as nn
import torch.nn.functional as F
from library.network_modules import ResidualBlock, ASPP
from library.config import Config


class CACResUNet(nn.Module):
    """
    Context-Aware Coordinate ResUNet (CAC-ResUNet).

    Architecture:
    - Encoder: Residual Blocks with Coordinate Attention.
    - Bottleneck: Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale context.
    - Decoder: Transposed Convolutions for upsampling + Residual Blocks.

    The network predicts the noise residual, which is subtracted from the input
    to obtain the denoised image.
    """

    def __init__(self):
        super(CACResUNet, self).__init__()

        filters = Config.BASE_FILTERS  # Typically 64
        in_channels = Config.IN_CHANNELS
        out_channels = Config.OUT_CHANNELS

        # --- Encoder ---
        # Initial convolution to map input to base filters
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.SiLU(),
        )

        # Level 1: Full resolution
        self.enc1 = ResidualBlock(filters, filters, stride=1)

        # Level 2: 1/2 resolution
        self.enc2 = ResidualBlock(filters, filters * 2, stride=2)

        # Level 3: 1/4 resolution
        self.enc3 = ResidualBlock(filters * 2, filters * 4, stride=2)

        # Level 4: 1/8 resolution
        self.enc4 = ResidualBlock(filters * 4, filters * 8, stride=2)

        # Level 5: 1/16 resolution (Bottleneck Input)
        self.enc5 = ResidualBlock(filters * 8, filters * 16, stride=2)

        # --- Bottleneck ---
        # ASPP to capture multi-scale context at the deepest level
        self.aspp = ASPP(filters * 16, filters * 16)

        # --- Decoder ---
        # Up 4: 1/16 -> 1/8
        self.up4 = nn.ConvTranspose2d(
            filters * 16, filters * 8, kernel_size=2, stride=2
        )
        self.dec4 = ResidualBlock(
            filters * 16, filters * 8, stride=1
        )  # Input: 8 (up) + 8 (skip) = 16

        # Up 3: 1/8 -> 1/4
        self.up3 = nn.ConvTranspose2d(filters * 8, filters * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(
            filters * 8, filters * 4, stride=1
        )  # Input: 4 (up) + 4 (skip) = 8

        # Up 2: 1/4 -> 1/2
        self.up2 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(
            filters * 4, filters * 2, stride=1
        )  # Input: 2 (up) + 2 (skip) = 4

        # Up 1: 1/2 -> Full
        self.up1 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(
            filters * 2, filters, stride=1
        )  # Input: 1 (up) + 1 (skip) = 2

        # --- Output Head ---
        self.out_conv = nn.Conv2d(filters, out_channels, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x0 = self.input_conv(x)

        e1 = self.enc1(x0)  # Shape: (B, 64, H, W)
        e2 = self.enc2(e1)  # Shape: (B, 128, H/2, W/2)
        e3 = self.enc3(e2)  # Shape: (B, 256, H/4, W/4)
        e4 = self.enc4(e3)  # Shape: (B, 512, H/8, W/8)
        e5 = self.enc5(e4)  # Shape: (B, 1024, H/16, W/16)

        # --- Bottleneck ---
        b = self.aspp(e5)  # Shape: (B, 1024, H/16, W/16)

        # --- Decoder Path ---

        # Block 4
        d4 = self.up4(b)  # Shape: (B, 512, H/8, W/8)
        # Handle potential padding issues if input dimensions weren't powers of 2
        if d4.size() != e4.size():
            d4 = F.interpolate(
                d4, size=e4.shape[2:], mode="bilinear", align_corners=False
            )
        d4 = torch.cat([d4, e4], dim=1)  # Shape: (B, 1024, H/8, W/8)
        d4 = self.dec4(d4)  # Shape: (B, 512, H/8, W/8)

        # Block 3
        d3 = self.up3(d4)  # Shape: (B, 256, H/4, W/4)
        if d3.size() != e3.size():
            d3 = F.interpolate(
                d3, size=e3.shape[2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e3], dim=1)  # Shape: (B, 512, H/4, W/4)
        d3 = self.dec3(d3)  # Shape: (B, 256, H/4, W/4)

        # Block 2
        d2 = self.up2(d3)  # Shape: (B, 128, H/2, W/2)
        if d2.size() != e2.size():
            d2 = F.interpolate(
                d2, size=e2.shape[2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e2], dim=1)  # Shape: (B, 256, H/2, W/2)
        d2 = self.dec2(d2)  # Shape: (B, 128, H/2, W/2)

        # Block 1
        d1 = self.up1(d2)  # Shape: (B, 64, H, W)
        if d1.size() != e1.size():
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, e1], dim=1)  # Shape: (B, 128, H, W)
        d1 = self.dec1(d1)  # Shape: (B, 64, H, W)

        # --- Output ---
        out = self.out_conv(d1)

        return out
