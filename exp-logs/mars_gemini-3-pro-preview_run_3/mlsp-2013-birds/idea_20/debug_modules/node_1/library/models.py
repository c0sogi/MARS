import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Factory function to instantiate a model architecture with a modified classification head.

    Uses the `timm` library to create models, ensuring consistent handling of
    pre-trained weights and input channel configurations.

    Args:
        model_name (str): Name of the architecture. Must be one of ['resnet18', 'efficientnet_b0', 'densenet121'].
        num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES (19).
        pretrained (bool): Whether to load ImageNet pre-trained weights. Defaults to True.

    Returns:
        torch.nn.Module: The initialized PyTorch model.

    Raises:
        ValueError: If the provided model_name is not in the supported list.
    """

    # Validate model name against the configuration
    if model_name not in Config.MODEL_ARCHITECTURES:
        raise ValueError(
            f"Invalid model name: '{model_name}'. "
            f"Supported architectures are: {Config.MODEL_ARCHITECTURES}"
        )

    print(f"Initializing model: {model_name}")
    print(f"  - Pretrained: {pretrained}")
    print(f"  - Input Channels: {Config.NUM_CHANNELS}")
    print(f"  - Output Classes: {num_classes}")

    try:
        # Create the model using timm
        # timm handles the replacement of the classification head (fc/classifier)
        # based on the num_classes argument.
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=Config.NUM_CHANNELS,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}' using timm.") from e

    return model
