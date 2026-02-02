import torch
import torch.nn as nn
import timm
from library import config


class WIVENet(nn.Module):
    """
    Weight-Inflated Volumetric Early-Fusion (WIVE) Network.

    Wraps an EfficientNet-B0 backbone and adapts the first convolutional layer
    to accept 9-channel volumetric inputs (3 depths x 3 modalities) by inflating
    pretrained ImageNet weights.
    """

    def __init__(self):
        super(WIVENet, self).__init__()

        # Use standard 3-channel pretrained backbone (Cite Lesson 00025)
        self.backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=True,
            num_classes=config.NUM_CLASSES,
            drop_rate=config.DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
