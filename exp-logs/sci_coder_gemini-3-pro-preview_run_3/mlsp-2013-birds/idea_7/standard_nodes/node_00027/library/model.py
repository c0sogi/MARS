import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet18(nn.Module):
    """
    Standard ResNet-18 model for bird species classification.
    Processes the entire resized spectrogram as a single image.
    """

    def __init__(self):
        super(BirdResNet18, self).__init__()

        # Load Pretrained ResNet18
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # Modify the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.backbone(x)
