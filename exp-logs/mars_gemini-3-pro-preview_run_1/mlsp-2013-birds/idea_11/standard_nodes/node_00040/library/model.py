import torch
import torch.nn as nn
from torchvision import models


class BirdResNet(nn.Module):
    """
    ResNet-34 based model for multi-label bird species classification.

    Attributes:
        backbone (nn.Module): The ResNet-34 backbone.
    """

    def __init__(self, num_classes=19, pretrained=True):
        """
        Initialize the BirdResNet model.

        Args:
            num_classes (int): Number of output classes (bird species). Default is 19.
            pretrained (bool): Whether to initialize with ImageNet weights. Default is True.
        """
        super(BirdResNet, self).__init__()

        # Load ResNet34 backbone
        # Using the modern weights API if available, otherwise fallback to pretrained=True
        if pretrained:
            try:
                weights = models.ResNet34_Weights.IMAGENET1K_V1
                self.backbone = models.resnet34(weights=weights)
            except AttributeError:
                # Fallback for older torchvision versions
                self.backbone = models.resnet34(pretrained=True)
        else:
            self.backbone = models.resnet34(weights=None)

        # The input to ResNet is expected to be 3 channels (RGB).
        # The data pipeline handles replicating the single-channel spectrogram
        # to 3 channels, so no modification to the first conv layer is needed.

        # Replace the final fully connected layer
        # ResNet's fc layer: (fc): Linear(in_features=512, out_features=1000, bias=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.backbone(x)
