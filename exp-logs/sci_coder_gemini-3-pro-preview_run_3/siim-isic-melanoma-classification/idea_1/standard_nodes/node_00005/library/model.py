import torch
import torch.nn as nn
import torchvision
from library.config import Config


class ISICModel(nn.Module):
    """
    EfficientNet-B0 based model for binary skin lesion classification.
    Switched from MobileNetV3 to EfficientNet-B0 for better feature extraction (Cite solution_lesson_node_00004).

    Architecture:
    - Backbone: EfficientNet-B0 (pre-trained on ImageNet)
    - Head: Global Average Pooling -> Dropout -> Linear(1) -> Logits
    """

    def __init__(self, pretrained=True):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet weights. Defaults to True.
        """
        super(ISICModel, self).__init__()

        # Load the pre-trained EfficientNet-B0 model
        weights = (
            torchvision.models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
        self.model = torchvision.models.efficientnet_b0(weights=weights)

        # The torchvision EfficientNet implementation consists of:
        # 1. self.features: The convolutional backbone
        # 2. self.avgpool: AdaptiveAvgPool2d(1)
        # 3. self.classifier: A Sequential block (Dropout -> Linear)

        # We need to replace the classifier head for our binary task.
        # For EfficientNet-B0, the classifier's Linear layer is at index 1.
        in_features = self.model.classifier[1].in_features

        # Replace the classifier with a single Linear layer to output logits.
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features, 1)
        )

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
