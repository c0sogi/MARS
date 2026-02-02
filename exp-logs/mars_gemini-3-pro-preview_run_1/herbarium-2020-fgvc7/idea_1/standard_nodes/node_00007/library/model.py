import torch
import torch.nn as nn
from torchvision import models
from library.config import NUM_CLASSES


class PlantClassifier(nn.Module):
    """
    A ResNet-50 based classifier for plant species identification.

    Attributes:
        model (torch.nn.Module): The underlying ResNet-50 model with a modified head.
    """

    def __init__(self, pretrained=True):
        """
        Initializes the PlantClassifier.

        Args:
            pretrained (bool): If True, loads ImageNet pre-trained weights. Defaults to True.
        """
        super(PlantClassifier, self).__init__()

        # Load the ResNet-50 backbone
        # Using the recommended Weights enum for torchvision >= 0.13
        # Switching to V2 weights for better transfer learning performance (Cite {solution_lesson_node_00006})
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V2
        else:
            weights = None

        self.model = models.resnet50(weights=weights)

        # Retrieve the number of input features for the final fully connected layer
        # For ResNet-50, this is typically 2048
        in_features = self.model.fc.in_features

        # Replace the final fully connected layer to match the number of classes in the dataset
        self.model.fc = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits of shape (batch_size, NUM_CLASSES).
        """
        return self.model(x)
