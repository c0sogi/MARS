import torch
import torch.nn as nn
import timm
from library.config import (
    MODEL_NAME,
    PRETRAINED,
    DROP_PATH_RATE,
    IN_CHANNELS,
    NUM_CLASSES,
)


class StabilizedStem(nn.Module):
    """
    A stabilized input stem that reduces high-density volumetric inputs (128 channels)
    to a standard feature depth (64 channels) using standard convolution for
    early global mixing.
    """

    def __init__(self, in_channels=128, out_channels=64):
        super(StabilizedStem, self).__init__()

        # Standard Convolution (groups=1) is critical here for mixing information
        # across all 32 slices and 4 modalities immediately.
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        """
        Explicit Kaiming/He Normal initialization to prevent gradient explosion
        when projecting from high-dimensional input space (128 channels).
        """
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MSSHDNetwork(nn.Module):
    """
    Modality-Structured Stabilized High-Density (MS-SHD) Network.

    Architecture:
    1. Input: (B, 128, 224, 224) - 32 slices * 4 modalities
    2. Stabilized Stem: Reduces to (B, 64, 112, 112) with global mixing
    3. Backbone: EfficientNet-B0 (in_chans=64)
    4. Head: GAP + Linear (via timm)
    """

    def __init__(self):
        super(MSSHDNetwork, self).__init__()

        # 1. Stabilized Global-Mixing Stem
        self.stem = StabilizedStem(in_channels=IN_CHANNELS, out_channels=64)

        # 2. Backbone
        # We configure EfficientNet to accept the 64-channel output from the stem.
        # drop_path_rate is used for regularization.
        # num_classes=1 creates the GAP + Linear head for binary classification.
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            in_chans=64,
            drop_path_rate=DROP_PATH_RATE,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x):
        # x shape: (B, 128, 224, 224)

        # Pass through Stabilized Stem
        # Output shape: (B, 64, 112, 112)
        x = self.stem(x)

        # Pass through Backbone + Head
        # Returns logits (B, 1)
        logits = self.backbone(x)

        return logits
