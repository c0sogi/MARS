import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from library.config import Config


class ContrailModel(nn.Module):
    """
    U-Net model with ConvNeXt-Small backbone for Contrail Identification.

    This model uses a pre-trained ConvNeXt-Small encoder from the `timm` library
    (via segmentation_models_pytorch). It is configured to accept 6 input channels,
    representing the Ash color scheme and temporal differences.
    """

    def __init__(self):
        super(ContrailModel, self).__init__()

        # Initialize U-Net architecture
        # encoder_name: 'convnext_small' as defined in Config
        # encoder_weights: 'imagenet' for transfer learning
        # in_channels: 6 (3 for Ash T_curr, 3 for Ash T_diff)
        # classes: 1 (Binary segmentation)
        # activation: None (Return raw logits for numerical stability with BCEWithLogitsLoss)
        self.model = smp.Unet(
            encoder_name=Config.BACKBONE,
            encoder_weights=Config.ENCODER_WEIGHTS,
            in_channels=Config.IN_CHANNELS,
            classes=1,
            activation=None,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 6, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1, Height, Width).
        """
        return self.model(x)
