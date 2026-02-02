import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet34(nn.Module):
    """
    A ResNet-34 based architecture for multi-label bird species classification.

    This model uses a standard ResNet-34 backbone initialized with ImageNet weights (optional).
    It expects 3-channel input (RGB), which is achieved via channel replication of the
    single-channel spectrograms during data preprocessing. The classification head is a
    simple linear layer projecting to the number of target species.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initializes the BirdResNet34 model.

        Args:
            pretrained (bool): Whether to load pre-trained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(BirdResNet34, self).__init__()

        # Determine the weights to load
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-34 backbone
        self.backbone = models.resnet34(weights=weights)

        # Replace the final fully connected layer.
        # The standard ResNet-34 fc layer has 512 input features.
        in_features = self.backbone.fc.in_features

        # We project directly to the number of classes (19).
        # No complex aggregation heads are used.
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch Size, 3, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch Size, Num Classes).
        """
        # Pass through the backbone (includes the modified fc layer)
        logits = self.backbone(x)
        return logits
