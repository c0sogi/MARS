import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name, pretrained=True):
    """
    Instantiates a PyTorch model using the timm library with a modified head for
    binary classification.

    This function ensures:
    1. The backbone is loaded (optionally with pretrained weights).
    2. Global Average Pooling (GAP) is used.
    3. Backbone-specific normalization layers (e.g., LayerNorm for ConvNeXt) are retained
       before the classifier to ensure stability.
    4. The classifier outputs a single scalar logit (num_classes=1).

    Args:
        model_name (str): The name of the architecture (e.g., 'convnext_tiny', 'tf_efficientnetv2_s').
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        nn.Module: The configured PyTorch model.
    """

    # Create the model using timm.
    # - num_classes=1: Configures the final linear layer to output a single logit.
    # - global_pool='avg': Enforces Global Average Pooling.
    # - in_chans=3: Explicitly sets input channels to RGB.
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=Config.NUM_CLASSES,
        in_chans=3,
        global_pool="avg",
    )

    return model
