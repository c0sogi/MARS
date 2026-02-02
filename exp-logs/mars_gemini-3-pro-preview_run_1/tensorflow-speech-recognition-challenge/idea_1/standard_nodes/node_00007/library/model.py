import torch
import torch.nn as nn
import timm
from library.config import Config


class SpectroCNN(nn.Module):
    """
    Wrapper for a ResNet-18 architecture using timm.
    Processes Log-Mel Spectrograms as 2D images.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, input_channels=1):
        """
        Args:
            num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES (12).
            input_channels (int): Number of input channels (1 for mono spectrograms).
        """
        super(SpectroCNN, self).__init__()

        # Use ResNet18 from timm
        # in_chans=1 adapts the first convolution layer
        self.model = timm.create_model(
            "resnet18",
            pretrained=False,
            in_chans=input_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input spectrograms of shape [Batch, 1, n_mels, time].

        Returns:
            torch.Tensor: Logits of shape [Batch, num_classes].
        """
        return self.model(x)
