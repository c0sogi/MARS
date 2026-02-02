import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Creates and returns a PyTorch model based on the specified architecture name.

    Uses the `timm` library to instantiate the model backbone. The classifier head
    is automatically modified to output a single logit (num_classes=1) suitable for
    binary classification with BCEWithLogitsLoss.

    Args:
        model_name (str): The name of the model architecture (e.g., 'convnext_tiny', 'densenet121').
                          Must be one of the architectures defined in Config.MODEL_ARCHS.
        pretrained (bool): Whether to load weights pretrained on ImageNet. Defaults to True.

    Returns:
        nn.Module: The initialized PyTorch model.

    Raises:
        ValueError: If the provided model_name is not supported in Config.MODEL_ARCHS.
    """

    # Validate that the requested model architecture is supported by our configuration
    if model_name not in Config.MODEL_ARCHS:
        raise ValueError(
            f"Model architecture '{model_name}' is not supported. "
            f"Available architectures: {Config.MODEL_ARCHS}"
        )

    # Instantiate the model using timm
    # num_classes=Config.NUM_CLASSES (1) ensures the final layer is a Linear(in_features, 1)
    # in_chans=3 ensures the input layer expects standard RGB images
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES, in_chans=3
    )

    return model
