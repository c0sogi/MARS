import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class BirdResNet(nn.Module):
    """
    A ResNet-34 based model for bird species classification.

    Attributes:
        backbone (torch.nn.Module): The ResNet-34 feature extractor.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Initializes the BirdResNet model.

        Args:
            num_classes (int): Number of output classes (bird species).
            pretrained (bool): Whether to use ImageNet pretrained weights.
        """
        super(BirdResNet, self).__init__()

        # Initialize ResNet-34 backbone
        if pretrained:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet34(weights=weights)

        # Replace the classification head
        # The original fc layer in ResNet-34 has 512 input features
        in_features = self.backbone.fc.in_features

        # Use a standard Linear Layer projecting to the classes
        # Strictly avoiding structural noise like Dropout as per lessons learned
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
                              Expects 1-channel spectrograms.

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        # Input Adaptation: Channel Replication
        # The backbone expects 3 input channels (RGB).
        # We copy the single-channel spectrogram to R, G, and B.
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return self.backbone(x)


def get_resnet34(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """
    Factory function to instantiate the BirdResNet model.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Use pretrained weights.

    Returns:
        BirdResNet: The instantiated model.
    """
    model = BirdResNet(num_classes=num_classes, pretrained=pretrained)
    return model
