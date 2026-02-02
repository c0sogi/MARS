import torch
import torch.nn as nn
import timm
from library.config import Config


def create_model(pretrained=True):
    """
    Initializes the ConvNeXt-Small model using the timm library.

    The function loads the architecture specified in Config.MODEL_NAME.
    It adapts the final classification head to match Config.NUM_CLASSES.

    Args:
        pretrained (bool): If True, loads weights pretrained on ImageNet.
                           Defaults to True.

    Returns:
        nn.Module: The configured PyTorch model moved to the appropriate device.
    """
    # Initialize the model using timm
    # passing num_classes automatically replaces the head with a Linear layer of the correct size
    model = timm.create_model(
        Config.MODEL_NAME, pretrained=pretrained, num_classes=Config.NUM_CLASSES
    )

    # Move the model to the specified device (GPU or CPU)
    model = model.to(Config.DEVICE)

    return model
