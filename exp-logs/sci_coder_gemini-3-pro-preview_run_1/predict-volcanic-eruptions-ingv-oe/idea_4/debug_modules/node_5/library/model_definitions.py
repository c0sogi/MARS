import torch
import torch.nn as nn
import timm
from library.config import Config


class SeismicEfficientNet(nn.Module):
    """
    A 2D-CNN architecture based on EfficientNet for seismic time-to-eruption prediction.

    This model takes 10-channel Log-Mel Spectrograms as input (one channel per sensor)
    and outputs a scalar regression prediction. It leverages transfer learning by
    initializing with ImageNet weights, adapting the first layer to the multi-channel
    input.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the SeismicEfficientNet model.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
                               The first layer weights are adapted for 10 channels.
        """
        super(SeismicEfficientNet, self).__init__()

        # Retrieve architecture settings from Config
        model_name = Config.CNN_MODEL_NAME  # e.g., 'tf_efficientnet_b0_ns'
        in_channels = Config.CNN_IN_CHANNELS  # 10 sensors

        # Create the backbone using timm
        # in_chans=10: Adapts the first convolutional layer (stem) to accept 10 channels.
        #              timm handles weight initialization (e.g., by averaging original RGB weights).
        # num_classes=1: Replaces the classification head with a linear layer outputting 1 value.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=in_channels, num_classes=1
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 10, Height, Width).
                              Represents the stacked spectrograms.

        Returns:
            torch.Tensor: Regression output of shape (Batch_Size, 1).
        """
        output = self.backbone(x)
        return output
