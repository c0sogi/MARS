import torch
import torch.nn as nn
import timm
from library.config import Config


def get_artwork_model(num_classes=Config.num_classes, pretrained=Config.pretrained):
    """
    Constructs the model for multi-label artwork attribute classification using timm.

    Uses resnet101d (ResNet101 with deep stem) for better feature extraction.
    Cite Lesson 10: Learning Rate Magnitude Outweighs Architectural Nuances (we keep high LR).
    Cite Lesson 5: Alleviating the Feature Bottleneck (we use deep backbone).

    Args:
        num_classes (int): The number of output classes (attributes).
        pretrained (bool): Whether to load ImageNet pretrained weights.

    Returns:
        torch.nn.Module: The configured model.
    """
    # Use timm to create the model
    # This handles the architecture (Deep Stem for 'd' variants) and head replacement
    model = timm.create_model(
        Config.model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
