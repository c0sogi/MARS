import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class CassavaResNet(nn.Module):
    """
    CassavaResNet model based on the ResNet-18 architecture.

    This model uses a ResNet-18 backbone, optionally pre-trained on ImageNet,
    and replaces the final classification head to match the number of classes
    in the Cassava Leaf Disease Classification task.
    """

    def __init__(
        self,
        pretrained: bool = Config.PRETRAINED,
        num_classes: int = Config.NUM_CLASSES,
    ):
        """
        Initialize the CassavaResNet model.

        Args:
            pretrained (bool): If True, loads weights pre-trained on ImageNet.
                               Defaults to Config.PRETRAINED.
            num_classes (int): The number of output classes.
                               Defaults to Config.NUM_CLASSES.
        """
        super(CassavaResNet, self).__init__()

        # Load the ResNeXt-50 backbone
        # We use the 'weights' parameter for newer torchvision versions
        if pretrained:
            weights = models.ResNeXt50_32X4D_Weights.DEFAULT
        else:
            weights = None

        self.model = models.resnext50_32x4d(weights=weights)

        # Replace the final fully connected layer
        # ResNet's final layer is named 'fc'
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw output logits with shape (B, num_classes).
        """
        return self.model(x)
