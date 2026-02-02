import torch
import torch.nn as nn
import timm
from library.config import (
    TOTAL_INPUT_CHANNELS,
    STEM_OUT_CHANNELS,
    BACKBONE_NAME,
    DROP_PATH_RATE,
)


class RMSHDStem(nn.Module):
    """
    Stabilized Global-Mixing Adapter (Stem).

    This module bridges the gap between high-density volumetric input (stacked as channels)
    and standard 2D backbones. It performs:
    1. Compression: Reduces 128 channels to 64.
    2. Early Fusion: Mixes information from all slices/modalities via standard Conv2d.
    3. Downsampling: Reduces spatial resolution by stride 2.
    4. Stabilization: Uses explicit He Normal initialization to prevent gradient explosion.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Standard Convolution (not grouped) for global mixing
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        """
        Explicit initialization to ensure stability with high-channel inputs.
        """
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class RMSHDNet(nn.Module):
    """
    Robust Modality-Structured High-Density (RMS-HD) Network.

    Architecture:
    1. Input: (B, 128, 224, 224) - 32 slices * 4 modalities
    2. Stem: Reduces to (B, 64, 112, 112)
    3. Backbone: EfficientNet-B0 (in_chans=64)
    4. Head: Global Avg Pool + Linear -> Logits
    """

    def __init__(self):
        super().__init__()

        # 1. Stabilized Stem
        self.stem = RMSHDStem(
            in_channels=TOTAL_INPUT_CHANNELS, out_channels=STEM_OUT_CHANNELS
        )

        # 2. Backbone
        # We use timm to create EfficientNet-B0.
        # in_chans=64 matches the output of our custom stem.
        # num_classes=1 creates the final linear layer for binary classification.
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=True,
            in_chans=STEM_OUT_CHANNELS,
            num_classes=1,
            drop_path_rate=DROP_PATH_RATE,
        )

    def forward(self, x):
        # Input: (B, 128, 224, 224)

        # Pass through Stabilized Stem
        # Output: (B, 64, 112, 112)
        x = self.stem(x)

        # Pass through Backbone (Feature Extraction + Pooling + Classifier)
        # Output: (B, 1)
        x = self.backbone(x)

        return x
