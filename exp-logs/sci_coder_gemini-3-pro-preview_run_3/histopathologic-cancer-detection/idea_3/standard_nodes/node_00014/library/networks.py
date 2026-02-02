import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to instantiate models for the pathology tumor detection task.
    Supports ResNet18 and ConvNeXt-Tiny architectures using timm.

    Args:
        model_name (str): The name of the architecture to use. Must be one of
                          Config.MODEL_ARCHITECTURES (e.g., 'resnet18', 'convnext_tiny').
        pretrained (bool): If True, loads weights pretrained on ImageNet. Defaults to True.

    Returns:
        nn.Module: The configured PyTorch model with a binary classification head (output size 1).
    """

    # Validate the requested architecture
    if model_name not in Config.MODEL_ARCHITECTURES:
        raise ValueError(
            f"Architecture '{model_name}' is not supported. "
            f"Available options: {Config.MODEL_ARCHITECTURES}"
        )

    # Create the model using timm
    # num_classes=Config.NUM_CLASSES (1) ensures the final head is set up for binary classification.
    # This automatically replaces the original ImageNet head (1000 classes) with a new Linear layer.
    # For ConvNeXt, this also correctly handles the preceding LayerNorm.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES, in_chans=3
    )

    return model
