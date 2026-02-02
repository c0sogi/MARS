import timm
import torch.nn as nn
from library.utils import seed_everything, get_device


def get_model(model_name: str, pretrained: bool = True):
    """
    Creates a model from the timm library with specific configurations for the
    Dog vs Cat classification task.

    Supported models:
    - resnet50: Uses 'resnet50.a1_in1k' weights (Standard CNN).
    - convnext_small: Uses 'convnext_small.fb_in1k' weights (Modern CNN).
    - maxvit_tiny: Uses 'maxvit_tiny_tf_224.in1k' weights (Multi-Axis Transformer).

    Args:
        model_name (str): Name of the model architecture to instantiate.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.

    Returns:
        nn.Module: The configured PyTorch model with a single output class (logit).
    """

    # Define mapping from logical names to specific timm model identifiers
    # as per the heterogeneous ensemble strategy.
    model_mapping = {
        "resnet50": "resnet50.a1_in1k",
        "convnext_small": "convnext_small.fb_in1k",
        "maxvit_tiny": "maxvit_tiny_tf_224.in1k",
    }

    if model_name not in model_mapping:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {list(model_mapping.keys())}"
        )

    timm_name = model_mapping[model_name]

    # Create the model using timm
    # num_classes=1 modifies the final head to output a single value (logit)
    # suitable for binary classification with BCEWithLogitsLoss.
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=1)

    return model
