import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import CSKBlock, ASPP


class CSKResUNet(nn.Module):
    """
    Coordinate Selective Kernel ResUNet (CSK-ResUNet).

    This architecture combines the U-Net structure with Selective Kernel (SK) attention
    and Coordinate Attention (CA) to adaptively capture features at multiple scales
    and preserve spatial precision.

    Global Residual Learning:
    The network predicts the noise residual. The clean image is obtained by:
    Clean_Image = Input_Image - Predicted_Noise
    """

    def __init__(self):
        super(CSKResUNet, self).__init__()

        filters = Config.BASE_FILTERS
        in_channels = Config.IN_CHANNELS
        out_channels = Config.OUT_CHANNELS

        # ==========================================
        # Stem
        # ==========================================
        # Initial feature extraction: 1 -> 64
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.SiLU(),
        )

        # ==========================================
        # Encoder (Downsampling Path)
        # ==========================================
        # Level 1: 64 -> 64 (Resolution: H x W)
        self.enc1 = CSKBlock(filters, filters, stride=1)

        # Level 2: 64 -> 128 (Resolution: H/2 x W/2)
        self.enc2 = CSKBlock(filters, filters * 2, stride=2)

        # Level 3: 128 -> 256 (Resolution: H/4 x W/4)
        self.enc3 = CSKBlock(filters * 2, filters * 4, stride=2)

        # Level 4: 256 -> 512 (Resolution: H/8 x W/8)
        self.enc4 = CSKBlock(filters * 4, filters * 8, stride=2)

        # ==========================================
        # Bridge
        # ==========================================
        # Multi-scale context aggregation at the bottleneck
        self.aspp = ASPP(filters * 8, filters * 8)

        # ==========================================
        # Decoder (Upsampling Path)
        # ==========================================
        # Level 4 Decoder: 512 -> 256
        self.up4 = nn.ConvTranspose2d(filters * 8, filters * 4, kernel_size=2, stride=2)
        # Input to block: 256 (from up) + 256 (from skip) = 512
        self.dec4 = CSKBlock(filters * 8, filters * 4, stride=1)

        # Level 3 Decoder: 256 -> 128
        self.up3 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=2, stride=2)
        # Input to block: 128 (from up) + 128 (from skip) = 256
        self.dec3 = CSKBlock(filters * 4, filters * 2, stride=1)

        # Level 2 Decoder: 128 -> 64
        self.up2 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)
        # Input to block: 64 (from up) + 64 (from skip) = 128
        self.dec2 = CSKBlock(filters * 2, filters, stride=1)

        # ==========================================
        # Head
        # ==========================================
        # Final projection to output channels (Noise Residual)
        self.final_conv = nn.Conv2d(filters, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal for Convolutions and
        Constant initialization for BatchNorm.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # --- Stem ---
        x_stem = self.stem(x)

        # --- Encoder ---
        s1 = self.enc1(x_stem)  # (B, 64, H, W)
        s2 = self.enc2(s1)  # (B, 128, H/2, W/2)
        s3 = self.enc3(s2)  # (B, 256, H/4, W/4)
        s4 = self.enc4(s3)  # (B, 512, H/8, W/8)

        # --- Bridge ---
        b = self.aspp(s4)  # (B, 512, H/8, W/8)

        # --- Decoder ---

        # Block 4
        d4_up = self.up4(b)  # (B, 256, H/4, W/4)
        # Safe interpolation to handle odd input dimensions
        if d4_up.size() != s3.size():
            d4_up = F.interpolate(
                d4_up, size=s3.shape[2:], mode="bilinear", align_corners=False
            )
        d4_cat = torch.cat([d4_up, s3], dim=1)
        d4 = self.dec4(d4_cat)

        # Block 3
        d3_up = self.up3(d4)  # (B, 128, H/2, W/2)
        if d3_up.size() != s2.size():
            d3_up = F.interpolate(
                d3_up, size=s2.shape[2:], mode="bilinear", align_corners=False
            )
        d3_cat = torch.cat([d3_up, s2], dim=1)
        d3 = self.dec3(d3_cat)

        # Block 2
        d2_up = self.up2(d3)  # (B, 64, H, W)
        if d2_up.size() != s1.size():
            d2_up = F.interpolate(
                d2_up, size=s1.shape[2:], mode="bilinear", align_corners=False
            )
        d2_cat = torch.cat([d2_up, s1], dim=1)
        d2 = self.dec2(d2_cat)

        # --- Output ---
        # Predict Noise Residual
        out = self.final_conv(d2)

        return out
