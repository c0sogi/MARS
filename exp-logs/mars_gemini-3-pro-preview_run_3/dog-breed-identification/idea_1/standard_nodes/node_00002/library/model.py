import torch
import torch.nn as nn
from torchvision import models


class ResNet50Baseline(nn.Module):
    """
    ResNet-50 based model for Dog Breed Classification.

    This class implements a ResNet-50 backbone with a custom fully connected head
    tailored for the specific number of dog breed classes. It supports loading
    pre-trained ImageNet weights for transfer learning.
    """

    def __init__(self, num_classes: int = 120, pretrained: bool = True):
        """
        Initializes the ResNet50Baseline model.

        Args:
            num_classes (int): The number of output classes (dog breeds). Defaults to 120.
            pretrained (bool): If True, loads weights pre-trained on ImageNet-1k. Defaults to True.
        """
        super(ResNet50Baseline, self).__init__()

        # Determine weights to load
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        # Load the ResNet-50 backbone
        self.backbone = models.resnet50(weights=weights)

        # Replace the final fully connected layer (fc)
        # The original fc layer has 2048 input features
        in_features = self.backbone.fc.in_features

        # Create new head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, num_classes)
        )

        # Initialize the new head
        # We use Xavier Uniform initialization for weights and Zero initialization for biases
        # This ensures the random head doesn't destabilize the pre-trained backbone initially
        nn.init.xavier_uniform_(self.backbone.fc[1].weight)
        if self.backbone.fc[1].bias is not None:
            nn.init.zeros_(self.backbone.fc[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)
