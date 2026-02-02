import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_name=Config.MODEL_NAME,
    num_classes=Config.NUM_CLASSES,
    pretrained=Config.PRETRAINED,
    drop_path_rate=Config.DROP_PATH_RATE,
    device=Config.DEVICE,
):
    """
    Initializes the ConvNeXt model using timm, modifies the head for the specific
    number of classes, and moves it to the specified device.

    Args:
        model_name (str): The name of the model architecture (default: 'convnext_tiny').
        num_classes (int): The number of output categories (default: 5270).
        pretrained (bool): Whether to load ImageNet-1k pretrained weights (default: True).
        drop_path_rate (float): Stochastic depth rate (default: 0.1).
        device (str): Computation device ('cuda' or 'cpu').

    Returns:
        torch.nn.Module: The initialized and configured model.
    """

    # Initialize the model using timm
    # passing num_classes tells timm to replace the head with one matching our target
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    # Move the model to the designated hardware
    model = model.to(device)

    return model
