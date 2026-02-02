import torch
import torch.nn as nn
import timm
import math
from library.config import Config


class Glioblastoma25D(nn.Module):
    """
    2.5D Stacked CNN.

    Treats multiple slices as channels in a single 2D image.
    Cite Lesson 00008: Simplification from Volumetric Sequence to 2D Stacking.
    """

    def __init__(self):
        super(Glioblastoma25D, self).__init__()

        # Input channels = Modalities * Slices
        in_chans = Config.NUM_CHANNELS * Config.NUM_SLICES

        # Backbone: EfficientNet-B0
        # num_classes=1 for direct binary classification
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED_BACKBONE,
            num_classes=1,
            in_chans=in_chans,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Total_Channels, Height, Width)
        Returns:
            logits: Output tensor of shape (Batch, 1)
        """
        return self.backbone(x)
