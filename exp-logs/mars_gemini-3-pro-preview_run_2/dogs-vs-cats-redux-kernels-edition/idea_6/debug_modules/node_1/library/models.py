import torch
import torch.nn as nn
import timm
from library.config import Config


def build_model(model_arch: str, pretrained: bool = True, num_classes: int = 1):
    """
    Builds and returns a neural network model using the timm library.

    This function instantiates the specified architecture (e.g., ConvNeXt or Swin Transformer)
    and configures the final classification head for the target task.

    Args:
        model_arch (str): The name of the model architecture to create.
                          Must be a valid model name in the timm library
                          (e.g., 'convnext_small.fb_in1k', 'swin_small_patch4_window7_224.ms_in1k').
        pretrained (bool): If True, loads weights pretrained on ImageNet. Defaults to True.
        num_classes (int): The number of output classes. Defaults to 1 for binary classification
                           (Dog vs Cat), which is suitable for use with BCEWithLogitsLoss.

    Returns:
        torch.nn.Module: The PyTorch model with the modified head.

    Raises:
        RuntimeError: If the model architecture name is invalid or not found in timm.
    """
    try:
        # Create the model using timm.
        # When num_classes is provided, timm automatically replaces the original classification
        # head (e.g., 1000 classes for ImageNet) with a new Linear layer having 'num_classes' outputs.
        model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create model architecture '{model_arch}': {e}")

    return model
