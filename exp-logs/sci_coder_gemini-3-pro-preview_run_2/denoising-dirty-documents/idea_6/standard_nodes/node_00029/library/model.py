import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import CoordinateAttention, ASPP, AttentionGate, ResidualBlock


class AG_CAC_ResUNet(nn.Module):
    """
    Attention-Gated Context-Aware Coordinate ResUNet (AG-CAC-ResUNet).

    Integrates:
    - Residual Blocks for deep gradient flow.
    - Coordinate Attention for precise positional feature extraction.
    - ASPP for multi-scale contextual modeling at the bottleneck.
    - Attention Gates to filter noise from skip connections.
    - Global Residual Learning (predicting noise).
    """

    def __init__(self):
        super(AG_CAC_ResUNet, self).__init__()

        # Configuration flags
        self.use_ca = Config.USE_COORDINATE_ATTENTION
        self.use_aspp = Config.USE_ASPP
        self.use_ag = Config.USE_ATTENTION_GATES

        base = Config.BASE_FILTERS

        # --- Initial Convolution ---
        self.inc = nn.Sequential(
            nn.Conv2d(Config.IN_CHANNELS, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.SiLU(),
        )

        # --- Encoder (Downsampling Path) ---
        # Level 1: (B, 64, H, W)
        self.enc1 = ResidualBlock(base, base, stride=1, use_ca=self.use_ca)

        # Level 2: (B, 128, H/2, W/2)
        self.enc2 = ResidualBlock(base, base * 2, stride=2, use_ca=self.use_ca)

        # Level 3: (B, 256, H/4, W/4)
        self.enc3 = ResidualBlock(base * 2, base * 4, stride=2, use_ca=self.use_ca)

        # Level 4: (B, 512, H/8, W/8)
        self.enc4 = ResidualBlock(base * 4, base * 8, stride=2, use_ca=self.use_ca)

        # --- Bridge (Bottleneck) ---
        # Downsample to H/16: (B, 1024, H/16, W/16)
        self.bridge_conv = ResidualBlock(
            base * 8, base * 16, stride=2, use_ca=self.use_ca
        )

        if self.use_aspp:
            self.aspp = ASPP(base * 16, base * 16)

        # --- Decoder (Upsampling Path) ---

        # Decoder 4: Process H/8 features
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        if self.use_ag:
            # Gate: 512, Skip: 512
            self.ag4 = AttentionGate(F_g=base * 8, F_l=base * 8, F_int=base * 4)
        self.dec4 = ResidualBlock(base * 16, base * 8, stride=1, use_ca=self.use_ca)

        # Decoder 3: Process H/4 features
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        if self.use_ag:
            # Gate: 256, Skip: 256
            self.ag3 = AttentionGate(F_g=base * 4, F_l=base * 4, F_int=base * 2)
        self.dec3 = ResidualBlock(base * 8, base * 4, stride=1, use_ca=self.use_ca)

        # Decoder 2: Process H/2 features
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        if self.use_ag:
            # Gate: 128, Skip: 128
            self.ag2 = AttentionGate(F_g=base * 2, F_l=base * 2, F_int=base)
        self.dec2 = ResidualBlock(base * 4, base * 2, stride=1, use_ca=self.use_ca)

        # Decoder 1: Process H features
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        if self.use_ag:
            # Gate: 64, Skip: 64
            self.ag1 = AttentionGate(F_g=base, F_l=base, F_int=base // 2)
        self.dec1 = ResidualBlock(base * 2, base, stride=1, use_ca=self.use_ca)

        # --- Output Head ---
        self.outc = nn.Conv2d(base, Config.OUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.inc(x)

        x1 = self.enc1(x0)  # Skip 1 (H, W)
        x2 = self.enc2(x1)  # Skip 2 (H/2, W/2)
        x3 = self.enc3(x2)  # Skip 3 (H/4, W/4)
        x4 = self.enc4(x3)  # Skip 4 (H/8, W/8)

        # --- Bridge ---
        b = self.bridge_conv(x4)  # (H/16, W/16)
        if self.use_aspp:
            b = self.aspp(b)

        # --- Decoder ---

        # Block 4
        d4 = self.up4(b)  # Upsample to H/8
        skip4 = x4
        if d4.shape[2:] != skip4.shape[2:]:
            d4 = F.interpolate(
                d4, size=skip4.shape[2:], mode="bilinear", align_corners=False
            )
        if self.use_ag:
            skip4 = self.ag4(g=d4, x=skip4)
        d4 = torch.cat([d4, skip4], dim=1)
        d4 = self.dec4(d4)

        # Block 3
        d3 = self.up3(d4)  # Upsample to H/4
        skip3 = x3
        if d3.shape[2:] != skip3.shape[2:]:
            d3 = F.interpolate(
                d3, size=skip3.shape[2:], mode="bilinear", align_corners=False
            )
        if self.use_ag:
            skip3 = self.ag3(g=d3, x=skip3)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.dec3(d3)

        # Block 2
        d2 = self.up2(d3)  # Upsample to H/2
        skip2 = x2
        if d2.shape[2:] != skip2.shape[2:]:
            d2 = F.interpolate(
                d2, size=skip2.shape[2:], mode="bilinear", align_corners=False
            )
        if self.use_ag:
            skip2 = self.ag2(g=d2, x=skip2)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.dec2(d2)

        # Block 1
        d1 = self.up1(d2)  # Upsample to H
        skip1 = x1
        if d1.shape[2:] != skip1.shape[2:]:
            d1 = F.interpolate(
                d1, size=skip1.shape[2:], mode="bilinear", align_corners=False
            )
        if self.use_ag:
            skip1 = self.ag1(g=d1, x=skip1)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1(d1)

        # --- Output ---
        noise_pred = self.outc(d1)

        # Global Residual Learning: Clean = Input - Noise
        return x - noise_pred
