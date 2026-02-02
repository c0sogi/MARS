import torch
import torch.nn as nn
import timm
from library.config import Config


class WIISNet(nn.Module):
    """
    Standard EfficientNet-B0 Wrapper.

    Simplified to use standard 3-channel input to leverage ImageNet pretrained weights
    effectively without custom inflation logic.
    Cite solution_lesson_node_00025
    """

    def __init__(self):
        super(WIISNet, self).__init__()

        # Load the pretrained EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
            in_chans=Config.INPUT_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
