import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class ResNet18Baseline(nn.Module):
    """
    ResNet-18 model for Breast Cancer Detection.

    This class implements a ResNet-18 architecture where the final fully connected
    layer is replaced to output a single scalar probability. This aligns with the
    Naive MIL (Multiple Instance Learning) approach, treating each mammogram image
    as an independent instance during the forward pass.

    Attributes:
        model (torchvision.models.resnet.ResNet): The ResNet-18 backbone.
        sigmoid (torch.nn.Sigmoid): Activation function to produce probabilities.
    """

    def __init__(self, pretrained=True):
        """
        Initialize the ResNet-18 Baseline model.

        Args:
            pretrained (bool): If True, initializes the backbone with ImageNet weights.
                               Defaults to True.
        """
        super(ResNet18Baseline, self).__init__()

        # Determine weights based on pretrained flag
        # Using the modern torchvision weights API
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-18 model
        self.model = models.resnet18(weights=weights)

        # Modify the final fully connected layer
        # Original ResNet-18 fc layer: Linear(in_features=512, out_features=1000)
        # We replace it with: Linear(in_features=512, out_features=1)
        # to output a single logit for binary classification.
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

        # Sigmoid activation to convert logit to probability [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).
                              The input is expected to be normalized and have 3 channels.

        Returns:
            torch.Tensor: Output probabilities of shape (Batch, 1).
        """
        # Pass input through the ResNet backbone and modified FC layer
        x = self.model(x)

        # Apply Sigmoid activation to get probability
        x = self.sigmoid(x)

        return x
