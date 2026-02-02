import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class CatheterModel(nn.Module):
    """
    MobileNetV3-Large based model for catheter detection.
    Replaces the default classifier with a single dense layer for multi-label classification.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(CatheterModel, self).__init__()

        # Load MobileNetV3 Large
        # Using DEFAULT weights which corresponds to the best available ImageNet weights
        if pretrained:
            weights = models.MobileNet_V3_Large_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.mobilenet_v3_large(weights=weights)

        # The torchvision MobileNetV3 implementation consists of:
        # features -> avgpool -> flatten -> classifier
        # We want to keep features and avgpool (GAP), but replace the classifier.

        # Get the input features of the original classifier.
        # The original classifier is a Sequential block, the first layer is Linear.
        # For MobileNetV3-Large, in_features is typically 960.
        in_features = self.backbone.classifier[0].in_features

        # Replace the classifier with a single Dense layer.
        # This satisfies the requirement: "Global Average Pooling layer followed immediately by a single dense layer"
        # The avgpool and flatten are handled in the backbone's forward method before the classifier.
        self.backbone.classifier = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W)

        Returns:
            torch.Tensor: Raw logits (B, NUM_CLASSES)
        """
        # The torchvision implementation of mobilenet_v3_large.forward does:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return self.backbone(x)
