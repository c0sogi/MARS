import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ResidualBlock(nn.Module):
    """
    Standard Residual Block for the ResNet Encoder.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> (+) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder Block for the U-Net pathway.
    Performs Upsampling -> Concatenation (Skip Connection) -> Convolutions.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        # The input to the conv layers will be the concatenation of the upsampled features
        # and the skip connection features.
        combined_channels = in_channels + skip_channels

        self.conv = nn.Sequential(
            nn.Conv2d(
                combined_channels, out_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # Upsample input features
        x_up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Concatenate with skip connection
        # Note: We assume dimensions match (32x32 and 16x16) due to padding in encoder
        out = torch.cat([x_up, skip], dim=1)

        # Process fused features
        out = self.conv(out)
        return out


class CactusResUNet(nn.Module):
    """
    Custom ResNet-UNet Architecture for Cactus Identification.

    Encoder: Lightweight ResNet (3 stages)
    Decoder: U-Net style upsampling with lateral skip connections
    Head: Global Average Pooling on high-res feature map -> Classifier
    """

    def __init__(self):
        super(CactusResUNet, self).__init__()

        # Channel configuration from Config: [16, 32, 64]
        c = Config.ENCODER_CHANNELS

        # --- Encoder (Backbone) ---
        # Initial convolution to scale up from RGB
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32 -> 32x32 (16 channels)
        self.layer1 = ResidualBlock(c[0], c[0], stride=1)

        # Stage 2: 32x32 -> 16x16 (32 channels)
        self.layer2 = ResidualBlock(c[0], c[1], stride=2)

        # Stage 3: 16x16 -> 8x8 (64 channels) - Bottleneck
        self.layer3 = ResidualBlock(c[1], c[2], stride=2)

        # --- Decoder (Upsampling path) ---
        # Decoder 1: Upsample 8x8 -> 16x16
        # Input: 64ch (Bottleneck), Skip: 32ch (Layer2), Output: 32ch
        self.decoder1 = DecoderBlock(
            in_channels=c[2], skip_channels=c[1], out_channels=c[1]
        )

        # Decoder 2: Upsample 16x16 -> 32x32
        # Input: 32ch (Decoder1), Skip: 16ch (Layer1), Output: 16ch
        self.decoder2 = DecoderBlock(
            in_channels=c[1], skip_channels=c[0], out_channels=c[0]
        )

        # --- Classification Head ---
        # Global Average Pooling on the recovered 32x32 feature map
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final dense layer
        self.fc = nn.Linear(c[0], Config.NUM_CLASSES)

    def forward(self, x):
        # --- Encoder ---
        x0 = self.initial_conv(x)  # [B, 16, 32, 32]

        e1 = self.layer1(x0)  # [B, 16, 32, 32] -> Skip connection 2
        e2 = self.layer2(e1)  # [B, 32, 16, 16] -> Skip connection 1
        e3 = self.layer3(e2)  # [B, 64, 8, 8]   -> Bottleneck

        # --- Decoder ---
        d1 = self.decoder1(e3, e2)  # [B, 32, 16, 16]
        d2 = self.decoder2(d1, e1)  # [B, 16, 32, 32]

        # --- Head ---
        # Apply GAP to the high-resolution feature map
        out = self.global_pool(d2)  # [B, 16, 1, 1]
        out = out.view(out.size(0), -1)  # Flatten
        logits = self.fc(out)  # [B, 1]

        return logits
