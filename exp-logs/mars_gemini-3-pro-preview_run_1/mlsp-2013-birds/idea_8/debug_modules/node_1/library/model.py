import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(device=None, pretrained=None):
    """
    Initializes the ResNet-34 model for bird species classification.

    Constructs a ResNet-34 backbone initialized with ImageNet weights (if requested),
    and replaces the final fully connected layer with a linear projection to the
    19 bird species classes.

    Args:
        device (str, optional): Device to move the model to ('cpu' or 'cuda').
                                Defaults to Config.DEVICE if not provided.
        pretrained (bool, optional): Whether to load ImageNet weights.
                                     Defaults to Config.PRETRAINED if not provided.

    Returns:
        torch.nn.Module: The configured ResNet-34 model.
    """
    if device is None:
        device = Config.DEVICE

    if pretrained is None:
        pretrained = Config.PRETRAINED

    # Load ResNet-34
    # Using the modern weights API for torchvision
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    model = models.resnet34(weights=weights)

    # Replace the fully connected layer
    # ResNet-34's fc layer input features is 512
    in_features = model.fc.in_features

    # As per the design, use a simple Linear Layer.
    # If dropout is specified in config, wrap in Sequential.
    if hasattr(Config, "DROPOUT") and Config.DROPOUT > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT), nn.Linear(in_features, Config.NUM_CLASSES)
        )
    else:
        model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    # Move the model to the specified device
    model = model.to(device)

    return model
