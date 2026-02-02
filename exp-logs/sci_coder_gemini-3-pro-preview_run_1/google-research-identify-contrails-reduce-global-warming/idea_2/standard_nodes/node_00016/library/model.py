import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block with skip connections.
    Performs upsampling, concatenation with skip connection, and convolution.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # Input to conv1 is the concatenation of upsampled input and skip connection
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, skip=None):
        # 1. Upsample the input from the deeper layer
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # 2. Concatenate skip connection
        if skip is not None:
            # Ensure spatial dimensions match (handle slight mismatches due to padding/pooling)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        # 3. Convolutions
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class ResNet34UNet(nn.Module):
    """
    U-Net architecture with a pre-trained ResNet34 encoder.
    """

    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(ResNet34UNet, self).__init__()

        # --- Encoder (ResNet34) ---
        # Use weights enum if available
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        except AttributeError:
            # Fallback for older torchvision versions or different environments
            weights = "IMAGENET1K_V1" if pretrained else None

        self.encoder = models.resnet34(weights=weights)

        # Handle non-standard input channels (e.g., if we weren't using 3-channel Ash composite)
        if in_channels != 3:
            # Replace the first convolution to accept in_channels
            # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.encoder.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Note: Pre-trained weights for the first layer are lost if in_channels != 3

        # --- Decoder ---
        # ResNet34 Channel Depths:
        # Layer 0 (relu): 64
        # Layer 1: 64
        # Layer 2: 128
        # Layer 3: 256
        # Layer 4: 512

        # Block 4 removed to limit downsampling to 16x (Cite solution_lesson_node_00015)
        # We stop at Layer 3 (256 channels, 16x downsampling)

        # Block 3: Upsample Layer 3 (256) -> Match Layer 2 (128)
        # Input to conv: 256 (upsampled) + 128 (skip)
        self.decoder3 = DecoderBlock(
            in_channels=256, skip_channels=128, out_channels=128
        )

        # Block 2: Upsample Block 3 (128) -> Match Layer 1 (64)
        self.decoder2 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)

        # Block 1: Upsample Block 2 (64) -> Match Layer 0 (64)
        self.decoder1 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Final Output Block
        # Upsample Block 1 (64) -> Original Size
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder Forward Pass ---
        # Input shape: (B, C, H, W)

        # Stem
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0 = self.encoder.relu(x)  # Skip 0: (B, 64, H/2, W/2)

        x = self.encoder.maxpool(x0)  # (B, 64, H/4, W/4)

        # ResNet Layers
        x1 = self.encoder.layer1(x)  # Skip 1: (B, 64, H/4, W/4)
        x2 = self.encoder.layer2(x1)  # Skip 2: (B, 128, H/8, W/8)
        x3 = self.encoder.layer3(x2)  # Bottleneck: (B, 256, H/16, W/16)
        # x4 removed

        # --- Decoder Forward Pass ---

        # Upsample x3, concat x2
        d3 = self.decoder3(x3, x2)  # -> (B, 128, H/8, W/8)

        # Upsample d3, concat x1
        d2 = self.decoder2(d3, x1)  # -> (B, 64, H/4, W/4)

        # Upsample d2, concat x0
        d1 = self.decoder1(d2, x0)  # -> (B, 64, H/2, W/2)

        # Final Upsample to original resolution
        # d1 is H/2, so we upsample by 2
        out = F.interpolate(
            d1, scale_factor=2, mode="bilinear", align_corners=True
        )  # -> (B, 64, H, W)
        out = self.final_conv(out)  # -> (B, out_channels, H, W)

        return out
