import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class AppleResNet34(nn.Module):
    """
    ResNet34 architecture for Apple Disease Detection.

    Attributes:
        backbone (torch.nn.Module): The ResNet34 backbone with the final layer modified.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): If True, loads ImageNet pretrained weights.
        """
        super(AppleResNet34, self).__init__()

        # Select weights based on configuration
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet34 backbone
        self.backbone = models.resnet34(weights=weights)

        # Replace the final fully connected layer
        # The original ResNet34 fc layer maps 512 features to 1000 classes (ImageNet).
        # We replace it to map to our specific number of classes (4).
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        return self.backbone(x)
