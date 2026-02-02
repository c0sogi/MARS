import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_identifier: str, pretrained: bool = True, num_classes: int = 1
) -> nn.Module:
    """
    Factory function to instantiate models using timm.
    Resolves model identifiers against Config.MODEL_CONFIGS to ensure
    the correct specific weights (e.g., 'resnet50.a1_in1k') are used.

    Args:
        model_identifier (str): The key from Config.MODEL_CONFIGS (e.g., 'resnet50', 'maxvit_tiny')
                                or a valid timm model name string.
        pretrained (bool): Whether to initialize with pretrained weights.
                           Defaults to True.
        num_classes (int): Number of output neurons. Defaults to 1 for binary classification.

    Returns:
        nn.Module: The PyTorch model ready for training or inference.
    """

    # 1. Resolve the specific timm model name from Config if a short key is provided
    if model_identifier in Config.MODEL_CONFIGS:
        model_name = Config.MODEL_CONFIGS[model_identifier]["model_name"]
    else:
        # Fallback: assume the identifier is already a valid timm model name
        model_name = model_identifier

    # 2. Create the model using timm
    # timm.create_model handles the replacement of the classification head
    # when num_classes is specified.
    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as e:
        raise ValueError(f"Failed to create model '{model_name}'. Error: {e}")

    return model
