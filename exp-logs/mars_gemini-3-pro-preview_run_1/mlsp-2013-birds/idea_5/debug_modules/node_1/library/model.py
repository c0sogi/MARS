import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet(nn.Module):
    """
    ResNet-34 based model for bird species classification.
    Replaces the final fully connected layer to match the number of species.
    """

    def __init__(
        self,
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        super(BirdResNet, self).__init__()

        # Load the ResNet-34 backbone
        # We use the modern 'weights' parameter for torchvision, falling back if needed
        if pretrained:
            self.backbone = models.resnet34(
                weights=models.ResNet34_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.resnet34(weights=None)

        # The input to the final FC layer in ResNet34 is 512 dimensions
        in_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        # If dropout is specified, add it before the final projection
        if dropout_rate > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, NumClasses)
        """
        return self.backbone(x)


def build_model(device=Config.DEVICE):
    """
    Factory function to create the model and move it to the configured device.

    Args:
        device (str): Device identifier ('cpu' or 'cuda').

    Returns:
        nn.Module: The initialized BirdResNet model.
    """
    model = BirdResNet(
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    )
    model.to(device)
    return model
