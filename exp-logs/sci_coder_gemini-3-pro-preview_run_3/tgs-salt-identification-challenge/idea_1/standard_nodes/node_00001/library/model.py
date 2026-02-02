import torch
import torch.nn as nn
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


class DecoderBlock(nn.Module):
    """
    LinkNet Decoder Block.
    Performs: Conv1x1 (reduce) -> TransposeConv3x3 (upsample) -> Conv1x1 (expand)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # LinkNet typically uses in_channels // 4 for internal width to save parameters
        mid_channels = max(in_channels // 4, 1)

        self.block = nn.Sequential(
            # 1x1 Conv to reduce dimensions
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # Transposed Conv to upsample (stride 2)
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # 1x1 Conv to expand dimensions to match the target skip connection
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class LinkNetResNet18(nn.Module):
    """
    LinkNet architecture with ResNet18 encoder.
    Designed for 128x128 input images.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super().__init__()

        # Load ResNet18 Encoder
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        # --- Encoder Layers ---
        # Initial block: Input (B, 3, 128, 128) -> Output (B, 64, 64, 64)
        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu)
        # MaxPool: Output (B, 64, 32, 32)
        self.maxpool = base.maxpool

        # ResNet Stages
        self.layer1 = base.layer1  # (B, 64, 32, 32)
        self.layer2 = base.layer2  # (B, 128, 16, 16)
        self.layer3 = base.layer3  # (B, 256, 8, 8)
        self.layer4 = base.layer4  # (B, 512, 4, 4)

        # --- Decoder Layers ---
        # LinkNet connects Encoder Layer N to Decoder Layer N via addition

        # dec4: Takes layer4 (512) -> Upsamples -> Adds layer3 (256)
        self.dec4 = DecoderBlock(in_channels=512, out_channels=256)

        # dec3: Takes dec4_out (256) -> Upsamples -> Adds layer2 (128)
        self.dec3 = DecoderBlock(in_channels=256, out_channels=128)

        # dec2: Takes dec3_out (128) -> Upsamples -> Adds layer1 (64)
        self.dec2 = DecoderBlock(in_channels=128, out_channels=64)

        # dec1: Takes dec2_out (64) -> Upsamples -> Adds in_block output (64)
        # Note: dec2_out is 32x32, in_block output is 64x64.
        self.dec1 = DecoderBlock(in_channels=64, out_channels=64)

        # --- Final Head ---
        # Upsample from 64x64 (dec1 out) to 128x128 (original input size)
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # --- Encoder Pass ---
        # Input: (B, 3, 128, 128)
        x_stem = self.in_block(x)  # (B, 64, 64, 64)
        x0 = self.maxpool(x_stem)  # (B, 64, 32, 32)

        x1 = self.layer1(x0)  # (B, 64, 32, 32)
        x2 = self.layer2(x1)  # (B, 128, 16, 16)
        x3 = self.layer3(x2)  # (B, 256, 8, 8)
        x4 = self.layer4(x3)  # (B, 512, 4, 4)

        # --- Decoder Pass ---
        # Block 4: Up(x4) + x3 -> 8x8
        d4 = self.dec4(x4) + x3

        # Block 3: Up(d4) + x2 -> 16x16
        d3 = self.dec3(d4) + x2

        # Block 2: Up(d3) + x1 -> 32x32
        d2 = self.dec2(d3) + x1

        # Block 1: Up(d2) + x_stem -> 64x64
        d1 = self.dec1(d2) + x_stem

        # --- Final Head ---
        # 64x64 -> 128x128
        out = self.final_deconv(d1)
        logits = self.final_conv(out)

        return logits
