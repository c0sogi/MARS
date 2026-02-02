import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(
    pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
):
    """
    Constructs the ResNet-34 model for multi-label bird species classification.

    Args:
        pretrained (bool): If True, loads weights pretrained on ImageNet.
        num_classes (int): Number of output classes (species).
        device (str): Device to move the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The configured ResNet-34 model.
    """
    # Initialize ResNet-34
    # We use the modern weights API if available, otherwise fallback to standard behavior
    if pretrained:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
    else:
        weights = None

    model = models.resnet34(weights=weights)

    # The input data is 3-channel (RGB), which matches ResNet's expected input.
    # The dataset class handles replicating the single-channel spectrogram to 3 channels.

    # Modify the classification head
    # ResNet-34's final layer is named 'fc' and has 512 input features.
    in_features = model.fc.in_features

    # Replace with a new Linear layer for our specific number of classes.
    # We avoid complex heads (like ConcatPooling) to prevent overfitting on the small dataset.
    model.fc = nn.Linear(in_features, num_classes)

    # Move the model to the specified device
    model = model.to(device)

    return model
