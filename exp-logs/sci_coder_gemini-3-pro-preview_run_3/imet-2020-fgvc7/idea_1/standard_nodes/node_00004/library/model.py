import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class ArtworkClassifier(nn.Module):
    """
    ResNet-34 based classifier for multi-label artwork classification.

    Attributes:
        model (torchvision.models.resnet.ResNet): The backbone and head of the network.
    """

    def __init__(
        self,
        num_classes: int = Config.NUM_CLASSES,
        pretrained: bool = Config.PRETRAINED,
    ):
        """
        Initializes the ArtworkClassifier.

        Args:
            num_classes (int): The number of output classes (attributes). Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to load ImageNet pre-trained weights. Defaults to Config.PRETRAINED.
        """
        super(ArtworkClassifier, self).__init__()

        # Determine weights based on the pretrained flag
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-34 backbone
        self.model = models.resnet34(weights=weights)

        # Replace the final fully connected layer
        # The original ResNet-34 fc layer has 512 input features
        in_features = self.model.fc.in_features

        # Create a new Linear layer mapping 512 features to num_classes
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, num_classes).
        """
        return self.model(x)
