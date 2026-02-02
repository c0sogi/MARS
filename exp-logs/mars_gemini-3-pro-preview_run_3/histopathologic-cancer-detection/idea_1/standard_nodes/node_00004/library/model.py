import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class TumorClassifier(nn.Module):
    """
    ResNet-18 based classifier for tumor detection in pathology patches.
    Replaces the final fully connected layer for binary classification.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights.
        """
        super(TumorClassifier, self).__init__()

        # Determine weights parameter based on pretrained flag
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None

        # Load the backbone
        self.model = models.resnet18(weights=weights)

        # The input images are 64x64 (after cropping).
        # ResNet reduces spatial dimensions by factor of 32 (2^5).
        # 64 / 32 = 2. The final feature map will be 512x2x2.
        # Global Average Pooling will reduce this to 512x1x1.
        # This is valid for the standard architecture.

        # Replace the final fully connected layer
        # ResNet18 fc input features is 512
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Pass through the modified ResNet
        # Output is (B, 1) because we replaced the FC layer
        return self.model(x)


def get_model() -> nn.Module:
    """
    Factory function to instantiate the model using global configuration.

    Returns:
        nn.Module: The initialized TumorClassifier.
    """
    model = TumorClassifier(pretrained=Config.PRETRAINED)
    return model
