import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class BirdResNet(nn.Module):
    """
    Spectral-Dynamic ResNet-34 for Bird Species Classification.

    This model uses a ResNet-34 backbone initialized with ImageNet weights.
    It is designed to accept 3-channel inputs constructed from the spectrogram
    and its first and second temporal derivatives (Deltas).

    Attributes:
        backbone (torch.nn.Module): The ResNet-34 feature extractor.
    """

    def __init__(self, num_classes, pretrained=True):
        """
        Initialize the BirdResNet model.

        Args:
            num_classes (int): The number of target bird species (classes).
            pretrained (bool): Whether to initialize the backbone with ImageNet weights.
                               Default is True.
        """
        super(BirdResNet, self).__init__()

        # Load ResNet-34 backbone
        # We use the modern 'weights' parameter for torchvision > 0.13
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = resnet34(weights=weights)

        # The dataset provides 3-channel inputs: [Intensity, Delta, Delta-Delta].
        # The standard ResNet first convolution layer (self.backbone.conv1)
        # accepts 3 input channels by default, so no modification is needed
        # for the input layer.

        # Replace the final Fully Connected (FC) layer
        # ResNet-34's final layer is named 'fc'.
        # We replace it with a simple Linear layer projecting to num_classes.
        # This adheres to the strategy of avoiding complex pooling heads.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).
                              Channels represent [Spectrogram, Delta, Delta-Delta].

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # The ResNet backbone handles the full forward pass:
        # Conv1 -> Bn1 -> Relu -> MaxPool -> Layers 1-4 -> AvgPool -> Flatten -> FC
        return self.backbone(x)
