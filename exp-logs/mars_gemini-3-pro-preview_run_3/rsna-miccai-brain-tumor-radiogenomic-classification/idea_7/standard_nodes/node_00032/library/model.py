import torch
import torch.nn as nn
import timm
from library import config


class MGMTNet(nn.Module):
    """
    2.5D Stacked CNN.

    Architecture:
    1. Input: (B, 64, 256, 256) - 64 channels from 4 modalities * 16 slices.
    2. Backbone: EfficientNet-B0 (modified input) -> Global Average Pooling -> Linear.
    3. Output: Single logit for binary classification.

    Cite Lesson 00030: Native Input Adaptation vs. Pre-Network Projection Layers
    """

    def __init__(self):
        super(MGMTNet, self).__init__()

        # Retrieve hyperparameters from config
        in_channels = config.IN_CHANNELS
        backbone_name = config.BACKBONE_NAME

        # Backbone
        # EfficientNet-B0 initialized with pretrained weights.
        # Modified to accept 'in_channels' (64) directly.
        # Output is set to 1 class (logit).
        # Cite Lesson 00015: Using Dropout and DropPath for regularization.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            in_chans=in_channels,
            num_classes=1,
            drop_rate=config.DROP_RATE,
            drop_path_rate=config.DROP_PATH_RATE,
        )

    def forward(self, x):
        # x shape: (B, 64, 256, 256)

        # Apply Backbone and Classification Head
        # Output shape: (B, 1)
        x = self.backbone(x)

        return x
