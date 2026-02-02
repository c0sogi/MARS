import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(pretrained=Config.PRETRAINED):
    """
    Constructs and returns the ResNet-18 model for Herbarium classification.

    Args:
        pretrained (bool): If True, loads weights pre-trained on ImageNet.

    Returns:
        model (torch.nn.Module): The PyTorch model.
    """
    # Determine weights based on pretrained flag
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # Load the base ResNet-18 model
    model = models.resnet18(weights=weights)

    # The input features for the final FC layer in ResNet-18 is 512
    in_features = model.fc.in_features

    # Replace the final fully connected layer
    # We use a single linear layer to map features to the number of classes
    model.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    return model
