import torch
import torch.nn as nn
from torchvision import models
from library.config import Config
from library.utils import seed_everything

# Set random seeds for reproducibility
seed_everything(Config.SEED)


class AppleResNet34(nn.Module):
    """
    ResNet34 model for Apple Disease Detection.

    Architecture:
    - Backbone: ResNet34 initialized with ImageNet weights.
    - Head: Global Average Pooling (inherent in ResNet) followed by a
      replaced Fully Connected layer mapping to the number of classes.
    """

    def __init__(self, pretrained=True):
        """
        Initializes the model.

        Args:
            pretrained (bool): If True, loads weights pretrained on ImageNet.
                               Defaults to True.
        """
        super(AppleResNet34, self).__init__()

        # Load ResNet34 backbone
        # Attempt to use the modern 'weights' API if available (torchvision >= 0.13)
        try:
            from torchvision.models import ResNet34_Weights

            weights = ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
        except ImportError:
            # Fallback for older torchvision versions
            self.backbone = models.resnet34(pretrained=pretrained)

        # Modify the head
        # The original FC layer in ResNet34 has 512 input features
        in_features = self.backbone.fc.in_features

        # Replace with a new Linear layer for our specific number of classes
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
