import torch
import torch.nn as nn
from library.config import Config
from library.modules import ResidualBlock, CoordinateAttention, ASPP, SubPixelUpsample


class CoSPResUNet(nn.Module):
    """
    Coordinate Sub-Pixel ResUNet (CoSP-ResUNet).

    A U-Net variant designed for image denoising that incorporates:
    1. Residual Blocks for feature extraction.
    2. ASPP for multi-scale bottleneck context.
    3. Sub-Pixel Convolution (PixelShuffle) for gradient-preserving upsampling.
    4. Coordinate Attention on skip connections to filter noise propagation.
    """

    def __init__(self):
        super(CoSPResUNet, self).__init__()

        filters = Config.BASE_FILTERS  # 64

        # --- Encoder ---
        # Initial Feature Extraction
        self.init_conv = nn.Sequential(
            nn.Conv2d(
                Config.NUM_CHANNELS, filters, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(filters),
            nn.SiLU(inplace=True),
        )

        # Level 1: 64 channels
        self.enc1 = ResidualBlock(filters, filters)
        self.pool1 = nn.MaxPool2d(2)

        # Level 2: 128 channels
        self.enc2 = ResidualBlock(filters, filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Level 3: 256 channels
        self.enc3 = ResidualBlock(filters * 2, filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        # Level 4: 512 channels
        self.enc4 = ResidualBlock(filters * 4, filters * 8)
        self.pool4 = nn.MaxPool2d(2)

        # --- Bottleneck ---
        # ASPP: 512 -> 1024 channels
        self.aspp = ASPP(filters * 8, filters * 16)

        # --- Decoder ---
        # Level 4: Upsample 1024 -> 512
        self.up4 = SubPixelUpsample(filters * 16, filters * 8)
        self.att4 = CoordinateAttention(filters * 8)
        # Concat (512 + 512) -> 1024 -> ResBlock -> 512
        self.dec4 = ResidualBlock(filters * 8 + filters * 8, filters * 8)

        # Level 3: Upsample 512 -> 256
        self.up3 = SubPixelUpsample(filters * 8, filters * 4)
        self.att3 = CoordinateAttention(filters * 4)
        # Concat (256 + 256) -> 512 -> ResBlock -> 256
        self.dec3 = ResidualBlock(filters * 4 + filters * 4, filters * 4)

        # Level 2: Upsample 256 -> 128
        self.up2 = SubPixelUpsample(filters * 4, filters * 2)
        self.att2 = CoordinateAttention(filters * 2)
        # Concat (128 + 128) -> 256 -> ResBlock -> 128
        self.dec2 = ResidualBlock(filters * 2 + filters * 2, filters * 2)

        # Level 1: Upsample 128 -> 64
        self.up1 = SubPixelUpsample(filters * 2, filters)
        self.att1 = CoordinateAttention(filters)
        # Concat (64 + 64) -> 128 -> ResBlock -> 64
        self.dec1 = ResidualBlock(filters + filters, filters)

        # --- Output Head ---
        # Projects to single channel noise residual
        self.final_conv = nn.Conv2d(filters, Config.NUM_CHANNELS, kernel_size=1)

    def forward(self, x):
        # --- Encoder Path ---
        x1 = self.init_conv(x)
        x1 = self.enc1(x1)  # Skip connection 1

        x2 = self.pool1(x1)
        x2 = self.enc2(x2)  # Skip connection 2

        x3 = self.pool2(x2)
        x3 = self.enc3(x3)  # Skip connection 3

        x4 = self.pool3(x3)
        x4 = self.enc4(x4)  # Skip connection 4

        # --- Bottleneck ---
        x_b = self.pool4(x4)
        x_b = self.aspp(x_b)

        # --- Decoder Path ---
        # Block 4
        d4 = self.up4(x_b)
        x4_att = self.att4(x4)  # Apply Coordinate Attention to skip connection
        d4 = torch.cat([d4, x4_att], dim=1)
        d4 = self.dec4(d4)

        # Block 3
        d3 = self.up3(d4)
        x3_att = self.att3(x3)
        d3 = torch.cat([d3, x3_att], dim=1)
        d3 = self.dec3(d3)

        # Block 2
        d2 = self.up2(d3)
        x2_att = self.att2(x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.dec2(d2)

        # Block 1
        d1 = self.up1(d2)
        x1_att = self.att1(x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.dec1(d1)

        # --- Output ---
        out = self.final_conv(d1)

        return out
