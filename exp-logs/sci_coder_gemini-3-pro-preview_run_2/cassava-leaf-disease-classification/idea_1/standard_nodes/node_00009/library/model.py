import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def get_model(num_classes, device):
    """
    Loads the ResNet-50 architecture with pre-trained ImageNet V2 weights,
    replaces the final fully connected layer to match the number of target classes,
    and moves the model to the specified device.
    Cite solution_lesson_node_00006: Synergistic Model Scaling with Label Smoothing and Enhanced Initialization.

    Args:
        num_classes (int): The number of output classes (target categories).
        device (str or torch.device): The device ('cpu' or 'cuda') to move the model to.

    Returns:
        torch.nn.Module: The modified ResNet-50 model ready for training/inference.
    """
    # Load the ResNet-50 model with ImageNet V2 weights
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)

    # The original ResNet-50 fc layer is: Linear(in_features=2048, out_features=1000, bias=True)
    # We retrieve the number of input features from the existing layer
    in_features = model.fc.in_features

    # Replace the final fully connected layer with a new one for our specific number of classes
    model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the specified computation device
    model = model.to(device)

    return model
