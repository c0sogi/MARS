import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleResNet34(nn.Module):
    """
    ResNet34 architecture for Apple Disease Detection.

    Structure:
    - Backbone: ResNet34 (pretrained on ImageNet)
    - Head: Global Average Pooling -> Linear Layer (4 classes)

    This simple architecture is chosen to avoid overfitting on the small dataset
    and serves as both the Teacher (in K-Fold) and Student (in Full-Data Distillation).
    """

    def __init__(self, num_classes, pretrained=True):
        super(AppleResNet34, self).__init__()

        # Load ResNet34 backbone
        # num_classes=0 removes the default classification head
        # global_pool='' removes the default pooling, returning spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            "resnet34", pretrained=pretrained, num_classes=0, global_pool=""
        )

        # ResNet34 typically has 512 output channels
        self.in_features = self.backbone.num_features

        # Explicit Global Average Pooling
        # Reduces (B, C, H, W) -> (B, C, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Simple Fully Connected Layer
        # Maps features to class logits
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        # Extract features from backbone
        x = self.backbone(x)

        # Pool features
        x = self.global_pool(x)

        # Flatten for linear layer: (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)

        # Classification
        logits = self.fc(x)

        return logits


def get_model(cfg, pretrained=True):
    """
    Factory function to initialize and return the AppleResNet34 model.

    Args:
        cfg (Config): Configuration object containing model and device parameters.
        pretrained (bool): Whether to load ImageNet weights.

    Returns:
        model (nn.Module): The initialized model moved to the specified device.
    """
    model = AppleResNet34(num_classes=cfg.num_classes, pretrained=pretrained)
    model.to(cfg.device)
    return model
