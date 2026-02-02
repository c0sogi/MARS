import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import random

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class UNetDecoderBlock(nn.Module):
    """
    U-Net Decoder Block.
    Performs: Upsample -> Concat -> Conv3x3 -> Conv3x3
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
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

    def forward(self, x, skip):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle size mismatch if any
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UNetResNet34(nn.Module):
    """
    U-Net architecture with ResNet34 encoder.
    Designed for 128x128 input images.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # Load ResNet34 Encoder
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        base = models.resnet34(weights=weights)

        # --- Encoder Layers ---
        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64x64
        self.maxpool = base.maxpool  # 32x32
        self.layer1 = base.layer1  # 32x32
        self.layer2 = base.layer2  # 16x16
        self.layer3 = base.layer3  # 8x8
        self.layer4 = base.layer4  # 4x4

        # --- Decoder Layers ---
        # dec4: In 512, Skip 256 -> Out 256
        self.dec4 = UNetDecoderBlock(512, 256, 256)

        # dec3: In 256, Skip 128 -> Out 128
        self.dec3 = UNetDecoderBlock(256, 128, 128)

        # dec2: In 128, Skip 64 -> Out 64
        self.dec2 = UNetDecoderBlock(128, 64, 64)

        # dec1: In 64, Skip 64 -> Out 64
        self.dec1 = UNetDecoderBlock(64, 64, 64)

        # --- Final Head ---
        # Upsample from 64x64 to 128x128
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1, bias=False
            ),  # 64->128
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # --- Encoder Pass ---
        x_stem = self.in_block(x)  # 64x64
        x0 = self.maxpool(x_stem)  # 32x32
        x1 = self.layer1(x0)  # 32x32
        x2 = self.layer2(x1)  # 16x16
        x3 = self.layer3(x2)  # 8x8
        x4 = self.layer4(x3)  # 4x4

        # --- Decoder Pass ---
        d4 = self.dec4(x4, x3)  # -> 8x8
        d3 = self.dec3(d4, x2)  # -> 16x16
        d2 = self.dec2(d3, x1)  # -> 32x32
        d1 = self.dec1(d2, x_stem)  # -> 64x64

        # --- Final Head ---
        logits = self.final_head(d1)  # -> 128x128

        return logits
