import torch
import torch.nn as nn
from library.config import Config


class DoubleConv(nn.Module):
    """
    Standard U-Net building block: (Conv2d => BN => ReLU) * 2
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


class DilatedBlock(nn.Module):
    """
    Dilated Bottleneck Block.
    Applies parallel convolutions with different dilation rates to capture multi-scale context
    without losing resolution or adding excessive parameters.
    """

    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4, 8]):
        super().__init__()

        # Ensure we can split channels evenly across branches
        num_branches = len(dilation_rates)
        if out_channels % num_branches != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by the number of dilation branches ({num_branches})"
            )

        branch_channels = out_channels // num_branches

        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
                for rate in dilation_rates
            ]
        )

        # 1x1 Convolution to fuse the concatenated features
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Apply each branch
        branch_outputs = [branch(x) for branch in self.branches]
        # Concatenate along channel dimension
        cat = torch.cat(branch_outputs, dim=1)
        # Fuse
        return self.fuse(cat)


class LeanUNet25D(nn.Module):
    """
    Lean 2.5D U-Net Architecture.

    Structure:
    1. Depth Compression: 3D (Z=65) -> 2D (C=32)
    2. Encoder: 3-stage downsampling (32 -> 64 -> 64)
    3. Bottleneck: Dilated Convolutions
    4. Decoder: 3-stage upsampling with skip connections
    5. Head: 1x1 Conv -> Sigmoid
    """

    def __init__(self):
        super().__init__()

        # --- Hyperparameters ---
        z_dim = Config.Z_DIM
        enc_channels = Config.ENCODER_CHANNELS  # Expected: [32, 64, 64]

        c1 = enc_channels[0]
        c2 = enc_channels[1]
        c3 = enc_channels[2]

        # --- 1. Depth Compression ---
        # Compresses the Z-dimension (65 slices) into a feature map (32 channels)
        # This acts as a learnable "soft" slice selection.
        self.depth_compression = nn.Sequential(
            nn.Conv2d(z_dim, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        # --- 2. Encoder ---
        self.enc1 = DoubleConv(c1, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        # --- 3. Dilated Bottleneck ---
        # Operates at the lowest resolution.
        # Uses dilation rates [1, 2, 4, 8] to expand receptive field.
        self.bottleneck = DilatedBlock(c3, c3, dilation_rates=[1, 2, 4, 8])

        # --- 4. Decoder ---

        # Stage 3 (Bottom)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels = c3 (from upsample) + c3 (from skip)
        # Output channels = c3
        self.dec3 = DoubleConv(c3 + c3, c3)

        # Stage 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels = c3 (from upsample) + c2 (from skip)
        # Output channels = c1
        self.dec2 = DoubleConv(c3 + c2, c1)

        # Stage 1 (Top)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Input channels = c1 (from upsample) + c1 (from skip)
        # Output channels = c1
        self.dec1 = DoubleConv(c1 + c1, c1)

        # --- 5. Head ---
        self.head = nn.Sequential(nn.Conv2d(c1, 1, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 65, Height, Width)
        Returns:
            Output tensor of shape (Batch, 1, Height, Width) with values in [0, 1]
        """

        # 1. Depth Compression
        x = self.depth_compression(x)  # (B, 32, H, W)

        # 2. Encoder
        e1 = self.enc1(x)  # (B, 32, H, W)
        p1 = self.pool1(e1)  # (B, 32, H/2, W/2)

        e2 = self.enc2(p1)  # (B, 64, H/2, W/2)
        p2 = self.pool2(e2)  # (B, 64, H/4, W/4)

        e3 = self.enc3(p2)  # (B, 64, H/4, W/4)
        p3 = self.pool3(e3)  # (B, 64, H/8, W/8)

        # 3. Bottleneck
        b = self.bottleneck(p3)  # (B, 64, H/8, W/8)

        # 4. Decoder
        # Up 3
        d3 = self.up3(b)  # (B, 64, H/4, W/4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)  # (B, 64, H/4, W/4)

        # Up 2
        d2 = self.up2(d3)  # (B, 64, H/2, W/2)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)  # (B, 32, H/2, W/2)

        # Up 1
        d1 = self.up1(d2)  # (B, 32, H, W)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)  # (B, 32, H, W)

        # 5. Head
        out = self.head(d1)  # (B, 1, H, W)

        return out
