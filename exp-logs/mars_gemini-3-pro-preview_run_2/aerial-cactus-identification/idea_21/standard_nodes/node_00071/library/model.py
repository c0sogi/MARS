import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config
from library.layers import ResNeStBlock


class CustomWideResNeSt(nn.Module):
    """
    Custom Wide Split-Attention Network (ResNeSt).

    Architecture:
    - Stem: Initial 3x3 Convolution.
    - Backbone: 3 Stages of ResNeSt Blocks with Split-Attention.
      - Stage 1: 32x32 resolution (Stride 1).
      - Stage 2: 16x16 resolution (Stride 2).
      - Stage 3: 8x8 resolution (Stride 2).
    - Head: Multi-Scale Aggregation.
      - Concatenates Global Average Pooled features from Stage 2 and Stage 3.
      - Single Dense Classification Layer.

    Configuration is derived from library.config.Config.
    """

    def __init__(self):
        super(CustomWideResNeSt, self).__init__()

        # Retrieve configuration
        input_channels = Config.INPUT_CHANNELS
        num_classes = Config.NUM_CLASSES
        stages_channels = Config.STAGES_CHANNELS  # [64, 128, 256]
        radix = Config.RADIX
        cardinality = Config.CARDINALITY

        # ==========================================
        # Stem
        # ==========================================
        self.stem = nn.Sequential(
            nn.Conv2d(
                input_channels,
                stages_channels[0],
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stages_channels[0]),
            nn.ReLU(inplace=True),
        )

        # ==========================================
        # Backbone (3 Stages)
        # ==========================================

        # Stage 1: 32x32 -> 32x32
        # Input: 64 -> Output: 64
        self.layer1 = ResNeStBlock(
            in_channels=stages_channels[0],
            out_channels=stages_channels[0],
            stride=1,
            radix=radix,
            cardinality=cardinality,
        )

        # Stage 2: 32x32 -> 16x16
        # Input: 64 -> Output: 128
        self.layer2 = ResNeStBlock(
            in_channels=stages_channels[0],
            out_channels=stages_channels[1],
            stride=2,
            radix=radix,
            cardinality=cardinality,
        )

        # Stage 3: 16x16 -> 8x8
        # Input: 128 -> Output: 256
        self.layer3 = ResNeStBlock(
            in_channels=stages_channels[1],
            out_channels=stages_channels[2],
            stride=2,
            radix=radix,
            cardinality=cardinality,
        )

        # ==========================================
        # Multi-Scale Aggregation Head
        # ==========================================
        # We concatenate features from Stage 2 (128 ch) and Stage 3 (256 ch)
        head_in_features = stages_channels[1] + stages_channels[2]

        self.fc = nn.Linear(head_in_features, num_classes)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Backbone
        x1 = self.layer1(x)  # Stage 1 Output (32x32)
        x2 = self.layer2(x1)  # Stage 2 Output (16x16)
        x3 = self.layer3(x2)  # Stage 3 Output (8x8)

        # Multi-Scale Aggregation
        # Apply Global Average Pooling to Stage 2 and Stage 3 outputs
        gap2 = F.adaptive_avg_pool2d(x2, 1).flatten(1)
        gap3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)

        # Concatenate features
        features = torch.cat([gap2, gap3], dim=1)

        # Classification
        out = self.fc(features)

        return out
