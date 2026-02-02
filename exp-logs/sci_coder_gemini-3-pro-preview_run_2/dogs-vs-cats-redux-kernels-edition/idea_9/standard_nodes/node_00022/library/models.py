import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Creates and returns a model based on the architecture name using the timm library.

    This function handles the instantiation of the backbone and modifies the
    classification head to output the specified number of classes (default is 1
    for binary classification).

    Args:
        model_name (str): Name of the model architecture (must be a valid timm model name).
        pretrained (bool): Whether to load pretrained weights (e.g., ImageNet).
        num_classes (int): Number of output classes. Defaults to 1 for binary classification.

    Returns:
        model (nn.Module): The instantiated PyTorch model with the modified head.
    """
    # Create the model using timm
    # passing num_classes automatically replaces the head with a Linear layer
    # having the correct number of outputs.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model
