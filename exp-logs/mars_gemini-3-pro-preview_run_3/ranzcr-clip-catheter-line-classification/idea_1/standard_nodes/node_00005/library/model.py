import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class CatheterResNet(nn.Module):
    """
    ResNet-34 based model for multi-label catheter detection.
    """

    def __init__(self, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
        """
        Args:
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            num_classes (int): Number of output classes (target labels).
        """
        super(CatheterResNet, self).__init__()

        # Select weights based on configuration
        if pretrained:
            weights = models.ResNet34_Weights.DEFAULT
        else:
            weights = None

        # Load the ResNet-34 backbone
        self.model = models.resnet34(weights=weights)

        # The dataset.py converts 1-channel X-rays to 3-channel RGB images.
        # Therefore, we keep the standard first convolution layer (accepts 3 channels).

        # Replace the final fully connected layer
        # ResNet-34's final layer is named 'fc' and has 512 input features.
        in_features = self.model.fc.in_features
        # Add Dropout to regularization (Cite solution_lesson_node_00004)
        self.model.fc = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        return self.model(x)
