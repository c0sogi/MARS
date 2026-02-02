import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.model_components import ResidualBlock


class ConvBlock(nn.Module):
    """
    Standard convolution block: Conv -> BN -> ReLU.
    Used for initial stem and feature reduction in decoder.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DecoderBlock(nn.Module):
    """
    Decoder block implementing:
    1. Bilinear Upsampling
    2. Convolution (channel reduction)
    3. Concatenation with encoder skip connection
    4. Residual Block with Coordinate Attention
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_reduce = ConvBlock(
            in_channels, in_channels // 2, kernel_size=1, padding=0
        )

        # Calculate input channels for the residual block after concatenation
        # Input: (Reduced Up-sampled Features) + (Skip Connection)
        block_in_channels = (in_channels // 2) + skip_channels

        self.res_block = ResidualBlock(
            in_channels=block_in_channels,
            out_channels=out_channels,
            use_ca=True,  # Coordinate Attention enabled in Decoder
            drop_path_rate=0.0,  # No stochastic depth in Decoder
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = self.conv_reduce(x)

        if skip is not None:
            # Ensure spatial dimensions match before concatenation (handling potential odd-size padding issues)
            if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
                x = F.interpolate(
                    x,
                    size=(skip.size(2), skip.size(3)),
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, skip], dim=1)

        return self.res_block(x)


class HighCapacityUNet(nn.Module):
    """
    High-Capacity Deep Residual U-Net with Coordinate Attention and Input Depth Fusion.
    """

    def __init__(self, drop_path_rate=0.2):
        super().__init__()

        # Input channels: Image (1) + Depth Map (1)
        in_channels = Config.CHANNELS + Config.DEPTH_CHANNELS

        # --- Encoder ---
        # Stem: 2 -> 64 (128x128)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder Blocks with Stochastic Depth
        # Layer 1: 64 -> 64 (128x128)
        self.enc1 = ResidualBlock(64, 64, stride=1, drop_path_rate=drop_path_rate)

        # Layer 2: 64 -> 128 (64x64)
        self.enc2 = ResidualBlock(64, 128, stride=2, drop_path_rate=drop_path_rate)

        # Layer 3: 128 -> 256 (32x32)
        self.enc3 = ResidualBlock(128, 256, stride=2, drop_path_rate=drop_path_rate)

        # Layer 4: 256 -> 512 (16x16)
        self.enc4 = ResidualBlock(256, 512, stride=2, drop_path_rate=drop_path_rate)

        # Bottleneck: 512 -> 1024 (8x8)
        self.center = ResidualBlock(512, 1024, stride=2, drop_path_rate=drop_path_rate)

        # --- Decoder ---
        # Dec 4: 1024 -> 512 (16x16)
        self.dec4 = DecoderBlock(1024, 512, 512)

        # Dec 3: 512 -> 256 (32x32)
        self.dec3 = DecoderBlock(512, 256, 256)

        # Dec 2: 256 -> 128 (64x64)
        self.dec2 = DecoderBlock(256, 128, 128)

        # Dec 1: 128 -> 64 (128x128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # --- Heads ---
        # Auxiliary heads for Deep Supervision
        self.head_32 = nn.Conv2d(256, 1, kernel_size=1)
        self.head_64 = nn.Conv2d(128, 1, kernel_size=1)

        # Final Output Head
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, images, depths):
        """
        Args:
            images: (B, 1, H, W) - Input images
            depths: (B,) or (B, 1) - Depth values (z)
        Returns:
            logits: (B, 1, H, W) - Final segmentation logits
        """
        b, c, h, w = images.shape

        # 1. Depth Fusion
        # Normalize depth (assuming range ~50-960, dividing by 1000 puts it in 0-1 range)
        z = depths.view(b, 1, 1, 1).float() / 1000.0
        # Expand scalar depth to a dense spatial feature map
        z_map = z.expand(b, 1, h, w)
        # Concatenate: (B, 1, H, W) + (B, 1, H, W) -> (B, 2, H, W)
        x = torch.cat([images, z_map], dim=1)

        # 2. Encoder Pass
        x = self.stem(x)  # (B, 64, H, W)
        e1 = self.enc1(x)  # (B, 64, H, W)
        e2 = self.enc2(e1)  # (B, 128, H/2, W/2)
        e3 = self.enc3(e2)  # (B, 256, H/4, W/4)
        e4 = self.enc4(e3)  # (B, 512, H/8, W/8)
        c = self.center(e4)  # (B, 1024, H/16, W/16)

        # 3. Decoder Pass
        d4 = self.dec4(c, e4)  # (B, 512, H/8, W/8)

        d3 = self.dec3(d4, e3)  # (B, 256, H/4, W/4)
        aux_32 = self.head_32(d3)  # Deep Supervision Head 1

        d2 = self.dec2(d3, e2)  # (B, 128, H/2, W/2)
        aux_64 = self.head_64(d2)  # Deep Supervision Head 2

        d1 = self.dec1(d2, e1)  # (B, 64, H, W)

        # 4. Final Output
        logits = self.final_conv(d1)

        if self.training:
            return logits, aux_64, aux_32
        else:
            return logits
