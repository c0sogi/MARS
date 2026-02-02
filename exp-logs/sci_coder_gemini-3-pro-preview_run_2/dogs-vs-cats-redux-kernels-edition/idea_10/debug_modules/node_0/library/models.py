import torch
import torch.nn as nn
import timm
from library.config import Config


def create_model(
    model_name: str, num_classes: int = Config.NUM_CLASSES, pretrained: bool = True
) -> nn.Module:
    """
    Factory function to create models based on the architecture name.
    Initializes specific timm backbones (ConvNeXt-Small and Swin-Small)
    with the correct number of output classes.

    Args:
        model_name (str): The name of the model architecture (must be one of Config.MODEL_NAMES).
        num_classes (int): The number of output classes (default: 1 for binary classification).
        pretrained (bool): Whether to load pretrained weights (default: True).

    Returns:
        nn.Module: The initialized PyTorch model.

    Raises:
        ValueError: If the provided model_name is not supported/configured.
    """

    # Validate model name against the configuration
    if model_name not in Config.MODEL_NAMES:
        raise ValueError(
            f"Model name '{model_name}' is not supported. "
            f"Available models: {Config.MODEL_NAMES}"
        )

    # Create the model using timm
    # timm handles the replacement of the classification head when num_classes is specified.
    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}' using timm.") from e

    return model
