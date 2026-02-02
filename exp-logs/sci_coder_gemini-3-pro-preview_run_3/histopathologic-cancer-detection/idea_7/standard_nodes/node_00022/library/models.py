import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_name: str,
    pretrained: bool = Config.PRETRAINED,
    num_classes: int = Config.NUM_CLASSES,
) -> nn.Module:
    """
    Factory function to initialize a timm model with a custom classifier head.

    Args:
        model_name (str): The name of the architecture to load (e.g., 'convnext_tiny').
        pretrained (bool): Whether to load pretrained ImageNet weights.
        num_classes (int): The number of output classes. Defaults to 1 for binary classification.
                           timm automatically replaces the head when this differs from the default.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    try:
        # Create the model using timm.
        # The library handles the download of weights and the modification of the
        # final classification layer (head) based on num_classes.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        return model

    except Exception as e:
        raise RuntimeError(f"Failed to initialize model '{model_name}': {str(e)}")
