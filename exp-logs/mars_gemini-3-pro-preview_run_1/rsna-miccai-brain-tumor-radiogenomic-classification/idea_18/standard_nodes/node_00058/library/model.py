import torch
import torch.nn as nn
import timm
from library.config import Config


class WIISNet(nn.Module):
    """
    Standard EfficientNet-B0 Baseline.
    Cite solution_lesson_node_00009: Avoid naive channel stacking of volumetric depth.
    Cite solution_lesson_node_00025: Deterministic slice selection is superior to learnable projections.
    """

    def __init__(self):
        super(WIISNet, self).__init__()

        # 1. Load Pretrained Backbone
        # Cite solution_lesson_node_00012: Use default dropout (implicit) rather than explicit override.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            in_chans=Config.IN_CHANNELS,  # Should be 3
        )

    def forward(self, x):
        return self.backbone(x)
