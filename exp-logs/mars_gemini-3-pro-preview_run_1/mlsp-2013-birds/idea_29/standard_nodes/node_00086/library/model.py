import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(
    device=Config.DEVICE, pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES
):
    """
    Constructs the ResNet-34 model for the ensemble distillation pipeline.

    Adheres to Idea 29 specifications:
    - Backbone: Vanilla ResNet-34 initialized with ImageNet weights.
    - Head: Standard Linear Layer (no Dropout, no complex aggregation).
    - Homogeneity: Used for both Teachers and Student.

    Args:
        device (str): Device to move the model to ('cpu' or 'cuda').
        pretrained (bool): Whether to load ImageNet weights.
        num_classes (int): Number of target bird species.

    Returns:
        torch.nn.Module: The configured ResNet-34 model.
    """
    # Select weights based on configuration
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None

    # Initialize the backbone
    model = models.resnet34(weights=weights)

    # Replace the classification head
    # The standard ResNet-34 fc layer has 512 input features
    in_features = model.fc.in_features

    # Replace with a simple Linear layer projecting to the number of classes
    # We strictly avoid Dropout or additional non-linearities here per the strategy
    model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the specified computing device
    model = model.to(device)

    return model
