import torch
import torch.nn as nn
from torchvision import models

try:
    from torchvision.models import ResNet50_Weights
except ImportError:
    ResNet50_Weights = None

from library.config import Config


def get_model(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
    """
    Constructs a ResNet-50 model adapted for the specific number of classes.

    Args:
        pretrained (bool): If True, loads weights pretrained on ImageNet.
        num_classes (int): The number of output categories.

    Returns:
        model (torch.nn.Module): The PyTorch model.
    """

    # Handle loading of pretrained weights compatible with different torchvision versions
    if pretrained:
        if ResNet50_Weights is not None:
            weights = ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            # Fallback for older torchvision versions
            model = models.resnet50(pretrained=True)
    else:
        model = models.resnet50(weights=None)

    # The input features for the final FC layer in ResNet50 is 2048
    in_features = model.fc.in_features

    # Replace the fully connected layer
    model.fc = nn.Linear(in_features, num_classes)

    # Initialize the new layer
    # Good practice to initialize the new layer's weights while keeping the pretrained backbone
    nn.init.xavier_uniform_(model.fc.weight)
    if model.fc.bias is not None:
        nn.init.constant_(model.fc.bias, 0)

    return model
