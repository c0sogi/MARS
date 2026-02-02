import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name, pretrained=True):
    """
    Initializes a model architecture using the timm library.

    The function loads the specified backbone, optionally loads pretrained ImageNet weights,
    and configures the final classification head for binary classification (1 output node).

    Args:
        model_name (str): The specific model identifier in the timm registry
                          (e.g., 'convnext_small.fb_in22k').
        pretrained (bool): If True, loads weights pretrained on ImageNet-21k/22k.
                           Defaults to True.

    Returns:
        torch.nn.Module: The instantiated PyTorch model ready for training or inference.
    """
    # Ensure the model name is valid within the context of the library,
    # though timm will raise its own error if the name is not found.
    # We do not strictly enforce Config.MODELS check here to allow flexibility
    # if the user wants to try other timm models without editing Config.

    print(f"Initializing model: {model_name}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Num Classes: 1 (Binary Classification)")

    try:
        # create_model handles the replacement of the classification head
        # (fc, classifier, head, etc.) automatically based on num_classes.
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,  # Binary classification (logit output for BCEWithLogitsLoss)
            in_chans=3,  # Standard RGB input
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create model '{model_name}' using timm. Details: {e}"
        )

    return model
