import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(pretrained=True):
    """
    Creates and returns the ResNet-34 model customized for the Herbarium 2020 task.

    Args:
        pretrained (bool): If True, use weights pre-trained on ImageNet.

    Returns:
        torch.nn.Module: The PyTorch model moved to the configured device.
    """
    # Determine weights parameter based on torchvision version and request
    if pretrained:
        weights = models.ResNet34_Weights.DEFAULT
    else:
        weights = None

    # Instantiate the ResNet-34 backbone
    model = models.resnet34(weights=weights)

    # Modify the final fully connected layer
    # ResNet-34's final layer is named 'fc' and has 512 input features
    in_features = model.fc.in_features

    # Replace the head with a new Linear layer for the specific number of classes
    # No dropout is added here as per Config.DROPOUT_RATE being 0.0,
    # but the architecture allows for it if the config changes.
    if hasattr(Config, "DROPOUT_RATE") and Config.DROPOUT_RATE > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )
    else:
        model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    # Move the model to the specified device (GPU/CPU)
    model = model.to(Config.DEVICE)

    return model
