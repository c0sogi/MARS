import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class CatheterModel(nn.Module):
    """
    EfficientNet-B0 based model for catheter detection.
    Replaces the default classifier with a single dense layer for multi-label classification.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(CatheterModel, self).__init__()

        # Load EfficientNet B0
        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.efficientnet_b0(weights=weights)

        # The torchvision EfficientNet implementation consists of:
        # features -> avgpool -> flatten -> classifier
        # The classifier is a Sequential block: (Dropout, Linear)

        # Get the input features of the original classifier.
        # For EfficientNet, the Linear layer is at index 1 of the classifier Sequential block.
        in_features = self.backbone.classifier[1].in_features

        # Replace the classifier with a single Dense layer.
        # This satisfies the requirement: "Global Average Pooling layer followed immediately by a single dense layer"
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
