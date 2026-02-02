import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
from library.config import Config


def get_mobilenet_model(
    pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
):
    """
    Constructs a MobileNetV3-Large model customized for the iNaturalist dataset.

    Args:
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
        num_classes (int): The number of output classes (species).
        device (str): The device to put the model on ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The modified MobileNetV3 model.
    """
    # Select weights
    if pretrained:
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    else:
        weights = None

    # Load the base model architecture
    model = models.mobilenet_v3_large(weights=weights)

    # Modify the classifier head
    # MobileNetV3 classifier is a Sequential block:
    # (0): Linear(in_features=960, out_features=1280)
    # (1): Hardswish()
    # (2): Dropout()
    # (3): Linear(in_features=1280, out_features=1000) -> Target for replacement

    # Get the input features of the final layer
    last_layer_index = len(model.classifier) - 1
    in_features = model.classifier[last_layer_index].in_features

    # Replace the last layer with one matching our number of classes
    model.classifier[last_layer_index] = nn.Linear(in_features, num_classes)

    # Move the model to the specified device
    model = model.to(device)

    return model
