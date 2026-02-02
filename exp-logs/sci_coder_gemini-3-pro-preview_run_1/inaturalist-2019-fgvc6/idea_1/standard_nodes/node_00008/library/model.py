import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
from library.config import Config


def get_model(
    pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
):
    """
    Constructs an EfficientNet-B0 model customized for the iNaturalist dataset.
    Cite solution_lesson_node_00006: Switch to EfficientNet for better accuracy/efficiency trade-off.

    Args:
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
        num_classes (int): The number of output classes (species).
        device (str): The device to put the model on ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The modified EfficientNet-B0 model.
    """
    # Select weights
    if pretrained:
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None

    # Load the base model architecture
    model = models.efficientnet_b0(weights=weights)

    # Modify the classifier head
    # EfficientNet-B0 classifier is a Sequential block:
    # (0): Dropout(p=0.2, inplace=True)
    # (1): Linear(in_features=1280, out_features=1000, bias=True) -> Target for replacement

    # Get the input features of the final layer
    last_layer_index = len(model.classifier) - 1
    in_features = model.classifier[last_layer_index].in_features

    # Replace the last layer with one matching our number of classes
    model.classifier[last_layer_index] = nn.Linear(in_features, num_classes)

    # Move the model to the specified device
    model = model.to(device)

    return model
