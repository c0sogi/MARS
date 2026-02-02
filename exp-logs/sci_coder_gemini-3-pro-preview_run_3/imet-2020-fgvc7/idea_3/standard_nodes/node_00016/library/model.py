import torch
import torch.nn as nn
import timm
from library.config import Config


def get_artwork_model(num_classes=Config.num_classes, pretrained=Config.pretrained):
    """
    Constructs the ResNet-101d model for multi-label artwork attribute classification.
    Cite solution_lesson_node_00014: Enhancing Feature Preservation with ResNet-D Deep Stems.

    Args:
        num_classes (int): The number of output classes (attributes).
        pretrained (bool): Whether to load ImageNet pretrained weights.

    Returns:
        torch.nn.Module: The configured ResNet-101d model.
    """
    # Use timm to create the model with Deep Stem configuration
    # This improves feature extraction for fine-grained details
    model = timm.create_model(
        "resnet101d", pretrained=pretrained, num_classes=num_classes
    )

    return model
