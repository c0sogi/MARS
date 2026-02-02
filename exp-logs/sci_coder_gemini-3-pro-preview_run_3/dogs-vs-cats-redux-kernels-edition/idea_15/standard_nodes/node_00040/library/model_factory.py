import torch
import torch.nn as nn
import timm
from library import config


def create_model(model_key, pretrained=True):
    """
    Creates a model instance based on the configuration key using the timm library.

    The function looks up the specific architecture name and parameters from
    config.MODEL_SPECS, initializes the model with pretrained weights (if requested),
    and configures the final classification head for binary classification (1 output node).

    Args:
        model_key (str): The key corresponding to a model in config.MODEL_SPECS
                         (e.g., 'resnet', 'convnext', 'maxvit').
        pretrained (bool): Whether to load pretrained ImageNet weights. Defaults to True.

    Returns:
        torch.nn.Module: The initialized PyTorch model.

    Raises:
        ValueError: If model_key is not found in the configuration.
    """
    if model_key not in config.MODEL_SPECS:
        raise ValueError(
            f"Invalid model_key '{model_key}'. "
            f"Available keys: {list(config.MODEL_SPECS.keys())}"
        )

    # Retrieve specific architecture name from config
    specs = config.MODEL_SPECS[model_key]
    timm_name = specs["timm_name"]

    # Create the model using timm
    # num_classes=1 sets the final layer to output a single logit for binary classification
    # (BCEWithLogitsLoss). timm automatically handles the replacement of the head.
    try:
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{timm_name}' using timm: {e}")

    return model
