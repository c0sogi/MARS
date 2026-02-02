import torch
import torch.nn as nn
from torchvision import models
from library import config


class BirdResNet(nn.Module):
    """
    ResNet18-based model for Bird Species Classification.

    Features:
    - Backbone: ResNet18 (Pretrained on ImageNet) (Cite solution_lesson_node_00028)
    - Input: 3-Channel (Filtered Spectrogram RGB)
    - Head: Global Average Pooling + Dropout + Linear (Cite solution_lesson_node_00024)
    """

    def __init__(self, pretrained=config.PRETRAINED):
        super(BirdResNet, self).__init__()

        # Load Pretrained ResNet18
        weights = "DEFAULT" if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        # Feature Extractor: Keep layers up to avgpool
        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Determine input features for the head (512 for ResNet18)
        in_features = self.backbone.fc.in_features

        # Simple Classification Head
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(in_features, config.NUM_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits (Batch, Num_Classes).
        """
        # Extract features (Batch, 512, 1, 1)
        x = self.feature_extractor(x)

        # Flatten (Batch, 512)
        x = torch.flatten(x, 1)

        # Classification Head (Batch, Num_Classes)
        x = self.head(x)

        return x
