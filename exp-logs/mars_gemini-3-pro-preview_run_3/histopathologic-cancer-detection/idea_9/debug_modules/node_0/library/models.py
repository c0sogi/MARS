import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to instantiate deep learning models based on the provided architecture name.

    This function utilizes the `timm` library to create models with ImageNet pretrained weights
    and modifies the final classification head to output a single logit for binary classification.

    Args:
        model_name (str): The name of the model architecture. Must be one of the models
                          defined in Config.MODELS (e.g., 'convnext_tiny', 'densenet121').
        pretrained (bool): If True, loads weights pretrained on ImageNet. Defaults to True.

    Returns:
        nn.Module: The PyTorch model ready for training/inference.

    Raises:
        ValueError: If the provided `model_name` is not supported in `Config.MODELS`.
    """

    # Validate that the requested model is part of the approved strategy
    if model_name not in Config.MODELS:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models in Config: {Config.MODELS}"
        )

    # print(f"Creating model: {model_name} | Pretrained: {pretrained}")

    # Instantiate the model using timm
    # num_classes=1 sets the final layer to output a single value (logit)
    # in_chans=3 ensures the input layer expects RGB images
    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1, in_chans=3
        )
    except Exception as e:
        raise RuntimeError(f"Error creating model '{model_name}' via timm: {str(e)}")

    return model
