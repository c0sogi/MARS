import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class DecoderBlock(nn.Module):
    """
    Standard U-Net Decoder Block: Upsample -> Concat -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
        x = self.up(x)

        if skip is not None:
            # Handle potential shape mismatch due to rounding in pooling layers
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class ResNet18UNet(nn.Module):
    """
    ResNet18 U-Net (Lesson 00052: Prefer ResNet18 without DropBlock for this task).

    Architecture:
    - Backbone: ResNet18 (Pretrained)
    - Decoder: U-Net style with Bilinear Upsampling
    - Heads:
        1. Classification (GAP + Linear)
        2. Segmentation (Conv)
    """

    def __init__(self):
        super().__init__()

        # Load Pretrained Backbone
        backbone = models.resnet18(pretrained=Config.PRETRAINED)

        # --- Encoder ---
        # Stem (Standard 7x7 Conv)
        self.first_conv = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )  # Out: 64 ch, H/2

        self.maxpool = backbone.maxpool  # Out: H/4

        # Residual Groups
        self.layer1 = backbone.layer1  # Out: 64 ch, H/4
        self.layer2 = backbone.layer2  # Out: 128 ch, H/8
        self.layer3 = backbone.layer3  # Out: 256 ch, H/16
        self.layer4 = backbone.layer4  # Out: 512 ch, H/32

        # --- Decoder ---
        # d4: Up(layer4) + layer3 -> 256 ch
        self.dec4 = DecoderBlock(512, 256, 256)
        # d3: Up(dec4) + layer2 -> 128 ch
        self.dec3 = DecoderBlock(256, 128, 128)
        # d2: Up(dec3) + layer1 -> 64 ch
        self.dec2 = DecoderBlock(128, 64, 64)
        # d1: Up(dec2) + first_conv -> 32 ch
        self.dec1 = DecoderBlock(64, 64, 32)

        # --- Heads ---
        # Segmentation Head
        self.seg_head = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            ),  # H/2 -> H
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
        )

        # Classification Head (Study Level)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Linear(512, Config.NUM_CLASSES)

    def forward(self, x):
        # --- Encoder Forward ---
        x0 = self.first_conv(x)  # (B, 64, H/2, W/2)
        x_mp = self.maxpool(x0)  # (B, 64, H/4, W/4)

        x1 = self.layer1(x_mp)  # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)  # (B, 128, H/8, W/8)

        x3 = self.layer3(x2)  # (B, 256, H/16, W/16)

        x4 = self.layer4(x3)  # (B, 512, H/32, W/32)

        # --- Classification Forward ---
        cls_feat = self.global_pool(x4)
        cls_feat = torch.flatten(cls_feat, 1)
        cls_logits = self.cls_head(cls_feat)

        # --- Decoder Forward ---
        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        # --- Segmentation Forward ---
        seg_logits = self.seg_head(d1)

        return cls_logits, seg_logits
