import torch
import torch.nn as nn
import timm
from library.config import Config


class WIISNet(nn.Module):
    """
    Simplified 2.5D Network using Middle Slice.

    Architecture:
        - Backbone: EfficientNet-B0 (Pretrained on ImageNet)
        - Input: 3 Channels (FLAIR, T1wCE, T2w)
    """

    def __init__(self):
        super(WIISNet, self).__init__()

        # 1. Load Pretrained Backbone
        # Cite solution_lesson_node_00012: Use default dropout (remove explicit drop_rate)
        # to avoid constraining the model with arbitrary manual tuning.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            in_chans=Config.IN_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch, 3, H, W)
        Returns:
            logits: Tensor of shape (Batch, 1)
        """
        return self.backbone(x)
