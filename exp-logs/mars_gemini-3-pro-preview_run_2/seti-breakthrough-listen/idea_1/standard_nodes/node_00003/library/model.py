import torch
import torch.nn as nn
import timm
from library.config import Config


class TechnosignatureModel(nn.Module):
    """
    A ResNet18-based model adapted for Technosignature Detection.
    Uses a vertically stacked input representation to handle signal drift.
    """

    def __init__(self, model_name="resnet18", pretrained=True):
        super(TechnosignatureModel, self).__init__()

        # Use timm to create a ResNet18 model
        # We modify in_chans to match our input (1 channel)
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.NUM_CHANNELS,
            num_classes=1,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 1, 1638, 256)

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1)
        """
        return self.model(x)
