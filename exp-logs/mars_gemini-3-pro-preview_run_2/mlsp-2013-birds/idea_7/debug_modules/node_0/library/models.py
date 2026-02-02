import torch
import torch.nn as nn
from torchvision import models


class BirdClassifier(nn.Module):
    """
    A wrapper class for creating bird species classifiers using ResNet18 or DenseNet121 backbones.

    Attributes:
        base_model (nn.Module): The underlying backbone model (ResNet or DenseNet).
    """

    def __init__(self, backbone="resnet18", pretrained=True, num_classes=19):
        """
        Initializes the BirdClassifier.

        Args:
            backbone (str): The architecture name. Options: 'resnet18', 'densenet121'.
            pretrained (bool): If True, loads weights pretrained on ImageNet.
            num_classes (int): The number of target classes (bird species).
        """
        super(BirdClassifier, self).__init__()

        if backbone == "resnet18":
            # Load ResNet18
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.base_model = models.resnet18(weights=weights)

            # Replace the final fully connected layer
            # ResNet: avgpool -> flatten -> fc
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Linear(in_features, num_classes)

        elif backbone == "densenet121":
            # Load DenseNet121
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            self.base_model = models.densenet121(weights=weights)

            # Replace the final classifier
            # DenseNet: features -> relu -> avgpool -> flatten -> classifier
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Linear(in_features, num_classes)

        else:
            raise ValueError(
                f"Invalid backbone '{backbone}'. Supported: 'resnet18', 'densenet121'."
            )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, 3, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, num_classes).
        """
        return self.base_model(x)
