import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Instantiates a model using the timm library, adapted for the specific task.

    Args:
        model_name (str): The name of the model architecture (e.g., 'convnext_small.fb_in22k').
        pretrained (bool): Whether to use pretrained weights.
        num_classes (int): Number of output neurons (1 for binary classification).

    Returns:
        nn.Module: The PyTorch model with the modified classification head.
    """
    # Create model with timm
    # timm.create_model automatically handles the replacement of the classification head
    # (e.g., 'head' or 'fc') when num_classes is different from the pretrained default.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
