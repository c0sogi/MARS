import torch
import torch.nn as nn
import timm
from library.config import (
    BACKBONE_NAME,
    STEM_OUT_CHANNELS,
    DROP_PATH_RATE,
    TOTAL_INPUT_CHANNELS,
)


class StabilizedStem(nn.Module):
    """
    Stabilized Global-Mixing Stem.
    Compresses high-density volumetric input (128 channels) to a stable feature space (64 channels).
    Performs early fusion and downsampling while maintaining gradient stability via He initialization.
    """

    def __init__(self, in_channels, out_channels):
        super(StabilizedStem, self).__init__()
        # Layer: Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        # Explicitly initialized using Kaiming/He Normal initialization
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


class MNSHDNetwork(nn.Module):
    """
    Modality-Normalized Stabilized High-Density (MNS-HD) Network.
    A 2.5D CNN designed to ingest high-density volumetric data (128 slices) via a
    stabilized stem before processing with an EfficientNet backbone.
    """

    def __init__(self):
        super(MNSHDNetwork, self).__init__()

        # 1. Stabilized Global-Mixing Stem
        # Compresses 128 channels -> 64 channels
        self.stem = StabilizedStem(
            in_channels=TOTAL_INPUT_CHANNELS, out_channels=STEM_OUT_CHANNELS
        )

        # 2. Backbone
        # EfficientNet-B0 instantiated via timm
        # Configured with in_chans=64 to accept the stem's output
        # drop_path_rate=0.2 for regularization
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=True,
            in_chans=STEM_OUT_CHANNELS,
            drop_path_rate=DROP_PATH_RATE,
            num_classes=0,  # Remove default head to get features
            global_pool="",  # Return spatial feature maps
        )

        # Determine number of features from backbone
        # We pass a dummy input through the backbone to get the output channels dynamically
        # Input to backbone is output of stem: (B, 64, 112, 112)
        with torch.no_grad():
            dummy_input = torch.zeros(1, STEM_OUT_CHANNELS, 112, 112)
            features = self.backbone(dummy_input)
            num_features = features.shape[1]

        # 3. Head
        # Global Average Pooling followed by a single fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        # Input shape: (B, 128, 224, 224)

        # Stem: (B, 128, 224, 224) -> (B, 64, 112, 112)
        x = self.stem(x)

        # Backbone: (B, 64, 112, 112) -> (B, C, H, W)
        x = self.backbone(x)

        # Global Average Pooling: (B, C, 1, 1)
        x = self.global_pool(x)

        # Flatten: (B, C)
        x = torch.flatten(x, 1)

        # FC: (B, 1) - Logits
        logits = self.fc(x)

        return logits
