import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from library.config import Config


class UNetPlusPlus25D(nn.Module):
    """
    A 2.5D U-Net++ model for MRI segmentation.

    Wraps the segmentation_models_pytorch implementation with:
    - Backbone: EfficientNet-B4
    - Input: 3 channels (Slice i-1, i, i+1)
    - Output: 3 classes (Large Bowel, Small Bowel, Stomach)
    - Deep Supervision: Enabled for multi-scale loss calculation
    """

    def __init__(self):
        super(UNetPlusPlus25D, self).__init__()

        # Initialize the U-Net++ model
        # We use the configuration defined in library.config
        self.model = smp.UnetPlusPlus(
            encoder_name=Config.BACKBONE,  # efficientnet-b4
            encoder_weights=Config.ENCODER_WEIGHTS,  # imagenet
            in_channels=Config.IN_CHANNELS,  # 3 (2.5D stack)
            classes=Config.NUM_CLASSES,  # 3
            activation=None,  # Return logits
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of shape (N, 3, H, W).

        Returns:
            torch.Tensor or List[torch.Tensor]:
                If deep_supervision is False: Output logits of shape (N, Classes, H, W).
                If deep_supervision is True: List of output logits from different decoder levels.
        """
        return self.model(x)
