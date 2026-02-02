import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class HerbariumResNet(nn.Module):
    """
    ResNet-50 based model for Herbarium Plant Species Classification.
    Replaces the final fully connected layer to match the number of classes in the dataset.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Args:
            num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to use ImageNet pre-trained weights. Defaults to Config.PRETRAINED.
        """
        super(HerbariumResNet, self).__init__()

        # Determine weights parameter based on pretrained flag
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
        else:
            weights = None

        # Load the ResNet-50 backbone
        self.backbone = models.resnet50(weights=weights)

        # The input features for the final FC layer in ResNet-50 is 2048
        in_features = self.backbone.fc.in_features

        # Replace the final fully connected layer
        # We use a single linear layer as specified in the design
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        return self.backbone(x)
