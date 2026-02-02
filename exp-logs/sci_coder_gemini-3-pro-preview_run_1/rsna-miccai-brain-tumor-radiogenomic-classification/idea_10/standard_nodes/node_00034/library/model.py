import torch
import torch.nn as nn
import timm
from library.config import (
    MODEL_BACKBONE,
    PRETRAINED,
    DROP_RATE,
    NUM_CLASSES,
    MODALITIES,
    SLAB_DEPTH,
)
from library.utils import set_seed

# Ensure reproducibility upon module import
set_seed()


class EarlyFusionNet(nn.Module):
    """
    Early Fusion Network.

    Uses a single EfficientNet-B0 backbone to process a multi-channel input
    composed of stacked modalities.
    """

    def __init__(
        self,
        backbone_name=MODEL_BACKBONE,
        pretrained=PRETRAINED,
        drop_rate=DROP_RATE,
        num_classes=NUM_CLASSES,
    ):
        super(EarlyFusionNet, self).__init__()

        # Calculate input channels dynamically to match data pipeline (Cite debug_lesson_5)
        in_chans = len(MODALITIES) * SLAB_DEPTH

        # Single backbone taking calculated input channels
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
            drop_rate=drop_rate,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of stacked images (B, 3, H, W).
        """
        return self.backbone(x)
