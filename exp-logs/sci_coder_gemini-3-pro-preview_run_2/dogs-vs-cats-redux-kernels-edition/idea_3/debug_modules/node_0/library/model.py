import timm
import torch.nn as nn
from library import config


def create_model(
    model_name=config.MODEL_NAME,
    num_classes=config.NUM_CLASSES,
    pretrained=config.PRETRAINED,
):
    """
    Initializes the ConvNeXt-Tiny architecture using the timm library.

    Args:
        model_name (str): Name of the model architecture in timm. Defaults to config.MODEL_NAME.
        num_classes (int): Number of output classes. Defaults to config.NUM_CLASSES.
        pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to config.PRETRAINED.

    Returns:
        torch.nn.Module: The initialized model.
    """
    # Create the model using timm
    # This automatically handles loading pretrained weights and modifying the head
    # for the specified number of classes (1 for binary classification logit).
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
