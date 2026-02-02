import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_name: str = Config.MODEL_NAME,
    pretrained: bool = Config.PRETRAINED,
    num_classes: int = Config.NUM_CLASSES,
    drop_rate: float = Config.DROP_RATE,
    drop_path_rate: float = Config.DROP_PATH_RATE,
    device: str = Config.DEVICE,
) -> nn.Module:
    """
    Instantiates the neural network model using the timm library.

    Args:
        model_name (str): The name of the model architecture (e.g., 'convnext_tiny').
        pretrained (bool): Whether to load weights pretrained on ImageNet.
        num_classes (int): The number of output classes (1 for binary classification).
        drop_rate (float): Dropout rate for the classification head.
        drop_path_rate (float): Drop path (stochastic depth) rate for regularization.
        device (str): The device ('cpu' or 'cuda') to move the model to.

    Returns:
        nn.Module: The instantiated PyTorch model ready for training or inference.
    """
    # Create the model using timm
    # timm handles the replacement of the classification head when num_classes is specified
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    # Move the model to the specified device
    model = model.to(device)

    return model
