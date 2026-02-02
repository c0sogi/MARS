import torch
import torch.nn as nn
import timm
from library.config import Config


class VolcanoEfficientNet(nn.Module):
    """
    VolcanoEfficientNet architecture based on EfficientNet-B0.

    This model is designed to process 10-channel Log-Mel Spectrograms generated from
    seismic sensor arrays. It adapts the standard EfficientNet architecture to:
    1. Accept 10 input channels (one per sensor) instead of the standard 3 (RGB).
    2. Output a single scalar value representing the predicted time_to_eruption.
    """

    def __init__(self, pretrained: bool = True):
        """
        Initialize the VolcanoEfficientNet model.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
                               The first layer weights are adapted for 10 channels.
                               Defaults to True.
        """
        super(VolcanoEfficientNet, self).__init__()

        # Extract configuration parameters
        model_name = Config.CNN_PARAMS.get("model_name", "efficientnet_b0")
        in_channels = Config.CNN_PARAMS.get("in_channels", 10)
        num_classes = Config.CNN_PARAMS.get("num_classes", 1)

        # Create the model using timm
        # - in_chans: Modifies the first conv layer to accept 'in_channels' (10).
        #             timm handles weight recycling/averaging from the 3-channel pretrained weights.
        # - num_classes: Replaces the default classifier with a linear layer of size 'num_classes' (1).
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              Expected shape: (Batch, 10, Freq, Time).

        Returns:
            torch.Tensor: Prediction tensor of shape (Batch, 1).
        """
        return self.backbone(x)
