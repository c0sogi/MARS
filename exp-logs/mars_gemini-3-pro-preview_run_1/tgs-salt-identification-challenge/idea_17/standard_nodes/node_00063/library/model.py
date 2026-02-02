import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    Enhances important features by recalibrating feature maps spatially and channel-wise.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation (cSE)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation (sSE)
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent combination: Input * cSE + Input * sSE
        return x * self.cSE(x) + x * self.sSE(x)


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with 2 Convolutional layers, BatchNorm, and ReLU.
    Used in Encoder and Decoder.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        # If input shape/channels don't match output, use 1x1 conv to adjust shortcut
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder block performing Upsampling, Concatenation, Feature Processing, and Attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Bilinear upsampling to avoid checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Process concatenated features (Input + Skip)
        self.block = ResidualBlock(in_channels + skip_channels, out_channels)

        # Attention mechanism
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle slight dimension mismatches if any (robustness)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        x = self.scse(x)
        return x


class DeepResUNet(nn.Module):
    """
    Deep Residual U-Net with scSE Attention and Deep Supervision.
    """

    def __init__(self):
        super().__init__()

        filters = Config.ENCODER_FILTERS  # [64, 128, 256, 512]
        in_ch = Config.INPUT_CHANNELS  # 2 (Image + Depth)

        # --- Encoder ---
        # Initial processing
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_ch, filters[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder Stages
        # Stage 1: 128x128
        self.enc1 = ResidualBlock(filters[0], filters[0])
        # Stage 2: 64x64
        self.enc2 = ResidualBlock(filters[0], filters[1], stride=2)
        # Stage 3: 32x32
        self.enc3 = ResidualBlock(filters[1], filters[2], stride=2)
        # Stage 4 (Bridge): 16x16
        self.center = ResidualBlock(filters[2], filters[3], stride=2)

        # --- Decoder ---
        # Dec 3: 16x16 -> 32x32
        self.dec3 = DecoderBlock(filters[3], filters[2], filters[2])
        # Dec 2: 32x32 -> 64x64
        self.dec2 = DecoderBlock(filters[2], filters[1], filters[1])
        # Dec 1: 64x64 -> 128x128
        self.dec1 = DecoderBlock(filters[1], filters[0], filters[0])

        # --- Prediction Heads (Deep Supervision) ---
        self.head3 = nn.Conv2d(filters[2], 1, 1)  # Aux Head (32x32)
        self.head2 = nn.Conv2d(filters[1], 1, 1)  # Aux Head (64x64)
        self.head1 = nn.Conv2d(filters[0], 1, 1)  # Main Head (128x128)

        self.deep_supervision = Config.DEEP_SUPERVISION

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, 1, H, W)
            z: Depth tensor (B,) or (B, 1)
        """
        # --- Input Fusion ---
        # Normalize depth (assuming range ~0-1000) to 0-1 range
        z = z.view(-1, 1, 1, 1).float() / 1000.0
        # Expand depth to match image spatial dimensions
        z_map = z.expand(-1, 1, x.size(2), x.size(3))
        # Concatenate along channel dimension: (B, 2, H, W)
        x = torch.cat([x, z_map], dim=1)

        # --- Encoder Forward ---
        x = self.input_conv(x)

        e1 = self.enc1(x)  # 128x128
        e2 = self.enc2(e1)  # 64x64
        e3 = self.enc3(e2)  # 32x32
        c = self.center(e3)  # 16x16 (Bottleneck)

        # --- Decoder Forward ---
        d3 = self.dec3(c, e3)  # 32x32
        d2 = self.dec2(d3, e2)  # 64x64
        d1 = self.dec1(d2, e1)  # 128x128

        # --- Heads ---
        out1 = self.head1(d1)  # Main output

        if self.training and self.deep_supervision:
            out2 = self.head2(d2)
            out3 = self.head3(d3)
            # Return list of outputs for Deep Supervision Loss
            return [out1, out2, out3]
        else:
            # Return only main output for inference/validation
            return out1
