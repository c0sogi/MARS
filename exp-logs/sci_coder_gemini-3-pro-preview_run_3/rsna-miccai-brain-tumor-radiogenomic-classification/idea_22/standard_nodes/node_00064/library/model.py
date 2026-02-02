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


class MSSHDNetwork(nn.Module):
    """
    Modality-Structured Network (2.5D Stacked).

    Architecture:
    1. Input: (B, 64, 224, 224) - 16 slices * 4 modalities
    2. Backbone: EfficientNet-B0 (in_chans=64)
    3. Head: GAP + Linear (via timm)
    """

    def __init__(self):
        super(MSSHDNetwork, self).__init__()

        # Backbone
        # We configure EfficientNet to accept the 64-channel input directly.
        # drop_path_rate is used for regularization.
        # num_classes=1 creates the GAP + Linear head for binary classification.
        self.backbone = timm.create_model(
            MODEL_NAME,
            pretrained=PRETRAINED,
            in_chans=IN_CHANNELS,
            drop_path_rate=DROP_PATH_RATE,
            num_classes=NUM_CLASSES,
        )

    def forward(self, x):
        # x shape: (B, 64, 224, 224)
        # Pass through Backbone + Head
        # Returns logits (B, 1)
        logits = self.backbone(x)

        return logits
