import torch
import torch.nn as nn
import timm
from library.config import Config


def get_artwork_model(num_classes=Config.num_classes, pretrained=Config.pretrained):
    """
    Constructs the ResNet-101d model for multi-label artwork attribute classification.

    Args:
        num_classes (int): The number of output classes (attributes).
        pretrained (bool): Whether to load ImageNet pretrained weights.

    Returns:
        torch.nn.Module: The configured ResNet-101d model.
    """
    # Use timm to create the model
    # resnet101d uses deep stem (3x3 convolutions) instead of 7x7
    model = timm.create_model(
        Config.model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
