import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_artwork_model(num_classes=Config.num_classes, pretrained=Config.pretrained):
    """
    Constructs the ResNet-101 model for multi-label artwork attribute classification.

    Args:
        num_classes (int): The number of output classes (attributes).
        pretrained (bool): Whether to load ImageNet pretrained weights.

    Returns:
        torch.nn.Module: The configured ResNet-101 model.
    """
    # Determine weights parameter based on pretrained flag
    # Using 'DEFAULT' loads the best available weights for the version
    weights = "DEFAULT" if pretrained else None

    # Load the ResNet-101 backbone
    # This architecture provides the depth required to disentangle complex attributes
    model = models.resnet101(weights=weights)

    # The standard ResNet architecture ends with:
    # 1. Global Average Pooling (avgpool)
    # 2. Fully Connected layer (fc)

    # We retrieve the input feature dimension of the final layer
    # For ResNet-101, this is 2048
    in_features = model.fc.in_features

    # Replace the final fully connected layer
    # This acts as the linear projection from the 2048-dim feature space to class logits
    model.fc = nn.Linear(in_features, num_classes)

    return model
