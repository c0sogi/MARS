import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(
    model_name, pretrained=True, num_classes=Config.NUM_CLASSES, device=Config.DEVICE
):
    """
    Factory function to instantiate a model backbone with a modified classification head.

    Implements the backbone selection for the Heterogeneous Ensemble strategy:
    - ResNet-18: Balanced capacity.
    - EfficientNet-B0: Parameter efficiency.
    - DenseNet-121: Feature reuse and robustness.

    Args:
        model_name (str): The name of the model architecture (must be supported by timm).
        pretrained (bool): Whether to load ImageNet pre-trained weights.
        num_classes (int): Number of target classes (default: 19).
        device (str): Device to place the model on ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The model ready for training/inference.
    """

    # Validate that the requested model is part of the intended strategy,
    # though we allow others if explicitly requested.
    if model_name not in Config.ARCHITECTURES:
        # Warning could be logged here, but we proceed to allow flexibility
        pass

    try:
        # Create the model using timm
        # in_chans=3 is strictly enforced by the strategy (3-Channel Rule)
        # timm handles the replacement of the classification head (fc/classifier)
        # automatically based on num_classes.
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
        )
    except Exception as e:
        raise ValueError(
            f"Failed to create model '{model_name}'. Ensure it is a valid timm model name. Error: {e}"
        )

    # Move model to the specified device
    if device:
        model = model.to(device)

    return model
