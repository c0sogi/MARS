import torch
import torch.nn as nn
import timm
from library.config import Config


class VolcanoEfficientNet(nn.Module):
    """
    Neural network architecture for the vision branch (Branch B).

    This model utilizes a pre-trained EfficientNet-B0 backbone, modified to accept
    10-channel inputs (Log-Mel Spectrograms from 10 sensors) and output a single
    regression value (time_to_eruption).
    """

    def __init__(self, model_name=None, pretrained=True, in_channels=None):
        """
        Initialize the VolcanoEfficientNet.

        Args:
            model_name (str, optional): Name of the timm model to use. Defaults to Config.CNN_PARAMS['model_name'].
            pretrained (bool, optional): Whether to load pretrained weights. Defaults to True.
            in_channels (int, optional): Number of input channels (sensors). Defaults to Config.CNN_PARAMS['in_channels'].
        """
        super(VolcanoEfficientNet, self).__init__()

        # Use defaults from Config if arguments are not provided
        self.model_name = (
            model_name if model_name is not None else Config.CNN_PARAMS["model_name"]
        )
        self.in_channels = (
            in_channels if in_channels is not None else Config.CNN_PARAMS["in_channels"]
        )

        # Create the model using timm
        # - in_chans=10: Modifies the first convolutional layer to accept 10 channels.
        #   timm handles the weight initialization for these new channels (often by averaging RGB weights).
        # - num_classes=1: Replaces the classifier with a linear layer outputting a single value (Regression).
        self.backbone = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            in_chans=self.in_channels,
            num_classes=1,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 10, Freq_Bins, Time_Steps).
                              Represents stacked Log-Mel Spectrograms.

        Returns:
            torch.Tensor: Output tensor of shape (Batch, 1).
                          Represents the predicted (possibly scaled) time_to_eruption.
        """
        return self.backbone(x)
