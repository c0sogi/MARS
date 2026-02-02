import torch
import torch.nn as nn
import timm


def get_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Initializes and returns a PyTorch model with the specified architecture.

    This function uses the `timm` library to construct the model backbone. It automatically
    replaces the default global pooling and fully connected layer with a configuration
    matching the specified number of target classes.

    Args:
        model_name (str): The name of the architecture (e.g., 'resnet34', 'densenet121').
        num_classes (int): The number of output classes for the final fully connected layer.
        pretrained (bool): If True, initializes the backbone with weights pretrained on ImageNet.

    Returns:
        nn.Module: The configured PyTorch model ready for training or inference.
    """
    try:
        # Create the model using timm.
        # - pretrained=True downloads and loads ImageNet weights.
        # - num_classes=num_classes replaces the final head with a Linear(in_features, num_classes).
        # - timm handles the appropriate Global Average Pooling for the specific architecture.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as e:
        raise ValueError(f"Failed to create model '{model_name}': {e}")

    return model
