import torch
import torch.nn as nn
from torchvision import models
from library.config import NUM_CLASSES


class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 based classifier for Cdiscount product categorization.
    Replaces the final classification head to output logits for 5270 classes.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        """
        Args:
            num_classes (int): Number of output categories. Defaults to config value.
            pretrained (bool): Whether to load ImageNet weights. Defaults to True.
        """
        super(MobileNetV2Classifier, self).__init__()

        # Load MobileNetV2 with optional pre-trained weights
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)

        # The classifier in MobileNetV2 is a Sequential module:
        # (0): Dropout(p=0.2, inplace=False)
        # (1): Linear(in_features=1280, out_features=1000, bias=True)

        # We access the input features of the linear layer (typically 1280)
        in_features = self.backbone.classifier[1].in_features

        # Replace the final Linear layer to match our number of classes
        # We preserve the original Dropout layer at index 0
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        return self.backbone(x)
