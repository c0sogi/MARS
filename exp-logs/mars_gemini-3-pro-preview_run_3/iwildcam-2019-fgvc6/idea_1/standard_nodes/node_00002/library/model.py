import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES):
    """
    Constructs and returns a ResNet-18 model customized for the specific number of classes.

    Args:
        pretrained (bool): If True, initializes the model with weights pre-trained on ImageNet.
        num_classes (int): The number of output classes for the classification head.

    Returns:
        torch.nn.Module: The modified ResNet-18 model.
    """
    # Determine weights parameter based on pretrained flag
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
    else:
        weights = None

    # Load the base ResNet-18 model
    model = models.resnet18(weights=weights)

    # Replace the final fully connected layer
    # The original fc layer is: (fc): Linear(in_features=512, out_features=1000, bias=True)
    in_features = model.fc.in_features

    # Create a new Linear layer with the correct number of output classes
    model.fc = nn.Linear(in_features, num_classes)

    return model
