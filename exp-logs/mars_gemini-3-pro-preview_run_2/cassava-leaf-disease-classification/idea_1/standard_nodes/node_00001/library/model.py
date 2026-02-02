import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(num_classes, device):
    """
    Loads the ResNet-18 architecture with pre-trained ImageNet weights,
    replaces the final fully connected layer to match the number of target classes,
    and moves the model to the specified device.

    Args:
        num_classes (int): The number of output classes (target categories).
        device (str or torch.device): The device ('cpu' or 'cuda') to move the model to.

    Returns:
        torch.nn.Module: The modified ResNet-18 model ready for training/inference.
    """
    # Load the ResNet-18 model with ImageNet weights
    # Using the specific weights enum as recommended for newer torchvision versions
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # The original ResNet-18 fc layer is: Linear(in_features=512, out_features=1000, bias=True)
    # We retrieve the number of input features from the existing layer
    in_features = model.fc.in_features

    # Replace the final fully connected layer with a new one for our specific number of classes
    model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the specified computation device
    model = model.to(device)

    return model
