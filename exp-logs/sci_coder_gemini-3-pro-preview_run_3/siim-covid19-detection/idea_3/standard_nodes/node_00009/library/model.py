import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library import config


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block: Upsample -> Concat with Skip -> ConvBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        # Ensure dimensions match (handle rounding errors in downsampling/upsampling)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MultiTaskUNet(nn.Module):
    """
    Multi-Task U-Net with MobileNetV2 Encoder.
    Performs simultaneous Segmentation (Opacity Detection) and Classification (Study Label).
    """

    def __init__(self, num_study_classes=config.NUM_STUDY_CLASSES, pretrained=True):
        super(MultiTaskUNet, self).__init__()

        # 1. Encoder: MobileNetV2
        # Load weights safely across torchvision versions
        try:
            from torchvision.models import MobileNet_V2_Weights

            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.mobilenet_v2(weights=weights)
        except ImportError:
            self.backbone = models.mobilenet_v2(pretrained=pretrained)

        self.features = self.backbone.features

        # MobileNetV2 Feature Indices for Skip Connections
        # Based on standard implementation:
        # Index 1:  x1/2  (16 channels)
        # Index 3:  x1/4  (24 channels)
        # Index 6:  x1/8  (32 channels)
        # Index 13: x1/16 (96 channels)
        # Index 18: x1/32 (1280 channels) - Bottleneck
        self.skip_indices = [1, 3, 6, 13]

        # 2. Decoder Path
        # Bottleneck (1280) -> x1/16
        self.dec4 = DecoderBlock(in_channels=1280, skip_channels=96, out_channels=256)
        # x1/16 -> x1/8
        self.dec3 = DecoderBlock(in_channels=256, skip_channels=32, out_channels=128)
        # x1/8 -> x1/4
        self.dec2 = DecoderBlock(in_channels=128, skip_channels=24, out_channels=64)
        # x1/4 -> x1/2
        self.dec1 = DecoderBlock(in_channels=64, skip_channels=16, out_channels=32)

        # 3. Final Upsampling (x1/2 -> x1)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # 4. Heads

        # Segmentation Head: Outputs logits for binary mask
        # Note: Sigmoid activation should be applied during inference or via BCEWithLogitsLoss
        self.seg_head = nn.Conv2d(16, 1, kernel_size=1)

        # Classification Head: Global Average Pooling -> Linear
        # Note: Softmax activation should be applied during inference or via CrossEntropyLoss
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.class_head = nn.Linear(1280, num_study_classes)

    def forward(self, x):
        # --- Encoder ---
        skips = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.skip_indices:
                skips.append(x)

        # Current x is the bottleneck (x1/32, 1280 ch)
        bottleneck = x

        # --- Classification Branch ---
        # Global Average Pooling
        pooled = self.avg_pool(bottleneck).flatten(1)
        class_logits = self.class_head(pooled)

        # --- Segmentation Branch (Decoder) ---
        # Skips: [x1/2, x1/4, x1/8, x1/16]
        # We consume them in reverse order

        d4 = self.dec4(bottleneck, skips[3])  # Uses x1/16
        d3 = self.dec3(d4, skips[2])  # Uses x1/8
        d2 = self.dec2(d3, skips[1])  # Uses x1/4
        d1 = self.dec1(d2, skips[0])  # Uses x1/2

        # Upsample to original resolution
        final_features = self.final_up(d1)

        # Predict mask logits
        seg_logits = self.seg_head(final_features)

        return seg_logits, class_logits
