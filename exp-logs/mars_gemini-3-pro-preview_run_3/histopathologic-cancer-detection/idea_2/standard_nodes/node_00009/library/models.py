import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Instantiates a neural network model using the timm library.

    The function configures the model for binary classification by setting the
    number of output classes to 1. It supports the architectures defined in
    Config.MODELS (resnet18, tf_efficientnetv2_s, convnext_tiny).

    Args:
        model_name (str): The name of the architecture to instantiate (e.g., 'resnet18').
        pretrained (bool): If True, loads weights pretrained on ImageNet. Defaults to True.

    Returns:
        nn.Module: The PyTorch model with a modified head for binary classification (1 output logit).
    """
    # Verify that the requested model is one of the expected architectures
    # strictly based on the provided Config, though the function is generic enough for others.
    if model_name not in Config.MODELS:
        # We allow it but print a debug message if needed, or just proceed.
        # Given the strict task, we proceed but assume the caller uses Config.MODELS.
        pass

    try:
        # Create the model using timm
        # num_classes=1 ensures the final linear layer outputs a single value (logit)
        # in_chans=3 ensures the input layer expects RGB images
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1, in_chans=3
        )

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to instantiate model '{model_name}': {str(e)}")
