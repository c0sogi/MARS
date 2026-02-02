import torch
import torch.nn as nn
import timm
from library.config import Config


class VolumetricEfficientNet(nn.Module):
    """
    A 2.5D Convolutional Neural Network for MGMT promoter methylation prediction.
    Uses Early Fusion by stacking slices and modalities in the channel dimension.

    Architecture:
    - Backbone: EfficientNet-B0
    - Input: 64 channels (16 slices * 4 modalities)
    - Head: Single linear layer for binary classification.
    """

    def __init__(self):
        super(VolumetricEfficientNet, self).__init__()

        # Initialize the backbone
        # in_chans=64 adapts the first layer to accept our 16 slices * 4 modalities input.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=Config.INPUT_CHANNELS,
            num_classes=0,
            global_pool="avg",
        )

        # Get the number of output features from the backbone
        num_features = self.backbone.num_features

        # Classification Head
        self.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor. Shape (B, 64, H, W).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Extract features
        features = self.backbone(x)

        # Classification
        logits = self.fc(features)

        return logits
