import timm
import torch
import torch.nn as nn
from library.config import Config
from library.utils import get_device


def get_model(model_name, pretrained=True):
    """
    Creates and returns a model based on the configuration defined in library.config.Config.
    Uses the timm library to instantiate the architecture with specific pretrained weights.

    Args:
        model_name (str): The key corresponding to the model in Config.MODELS
                          (e.g., 'resnet50', 'convnext_small', 'maxvit_tiny').
        pretrained (bool): Whether to initialize the model with pretrained weights (default: True).

    Returns:
        torch.nn.Module: The instantiated PyTorch model, moved to the configured device.

    Raises:
        ValueError: If the provided model_name is not found in Config.MODELS.
    """
    # Validate model name against configuration
    if model_name not in Config.MODELS:
        available_models = list(Config.MODELS.keys())
        raise ValueError(
            f"Model '{model_name}' is not defined in Config.MODELS. Available options: {available_models}"
        )

    # Retrieve architecture specification from config
    model_config = Config.MODELS[model_name]
    arch = model_config["arch"]

    # Create the model using timm
    # We set num_classes=1 for binary classification (Dog vs Cat)
    # This sets up the final linear layer to output a single logit
    model = timm.create_model(arch, pretrained=pretrained, num_classes=1)

    # Move the model to the appropriate device (GPU/CPU)
    device = get_device()
    model.to(device)

    return model
