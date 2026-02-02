import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet(nn.Module):
    """
    ResNet-34 based model for Bird Species Classification.
    Initialized with ImageNet weights and adapted for 19-class multi-label classification.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights. Defaults to Config.PRETRAINED.
        """
        super(BirdResNet, self).__init__()

        # Determine weights based on configuration
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        # Load the ResNet-34 backbone
        self.backbone = models.resnet34(weights=weights)

        # Replace the fully connected head
        # ResNet-34's final layer is named 'fc' and has 512 input features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.backbone(x)
