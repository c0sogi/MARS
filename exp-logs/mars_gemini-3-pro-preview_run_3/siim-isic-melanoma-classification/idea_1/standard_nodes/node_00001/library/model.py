import torch
import torch.nn as nn
import torchvision
from library.config import Config


class ISICModel(nn.Module):
    """
    MobileNetV3-Small based model for binary skin lesion classification.

    Architecture:
    - Backbone: MobileNetV3-Small (pre-trained on ImageNet)
    - Head: Global Average Pooling -> Linear(1) -> Logits
    """

    def __init__(self, pretrained=True):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet weights. Defaults to True.
        """
        super(ISICModel, self).__init__()

        # Load the pre-trained MobileNetV3-Small model
        weights = (
            torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )
        self.model = torchvision.models.mobilenet_v3_small(weights=weights)

        # The torchvision MobileNetV3 implementation consists of:
        # 1. self.features: The convolutional backbone
        # 2. self.avgpool: AdaptiveAvgPool2d(1)
        # 3. self.classifier: A Sequential block (Linear -> Hardswish -> Dropout -> Linear)

        # We need to replace the classifier head for our binary task.
        # First, determine the input features to the classifier.
        # For MobileNetV3-Small, the last convolutional layer output is projected
        # to 576 channels before the classifier.
        # We can dynamically get this from the first layer of the existing classifier.
        in_features = self.model.classifier[0].in_features

        # Replace the classifier with a single Linear layer to output logits.
        # Note: We do not include Sigmoid here because the loss function
        # (BCEWithLogitsLoss) handles it for numerical stability.
        self.model.classifier = nn.Sequential(nn.Linear(in_features, 1))

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # The torchvision implementation of mobilenet_v3 forward pass is:
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        # return x

        # Since we modified self.classifier, we can just call the base model
        return self.model(x)
