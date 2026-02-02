import torch.nn as nn
import timm
from library.config import Config


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Instantiates a model architecture using the timm library.

    The function automatically modifies the classification head to output a single
    logit (num_classes=1) for binary classification.

    Args:
        model_name (str): The name of the model architecture (e.g., 'convnext_tiny.fb_in22k_ft_in1k').
        pretrained (bool): Whether to load pre-trained ImageNet weights. Defaults to True.

    Returns:
        nn.Module: The PyTorch model with a modified head.
    """

    # Create the model using timm
    # num_classes=1 configures the head for binary classification (outputting a single logit)
    # in_chans=3 specifies RGB input
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES, in_chans=3
    )

    return model
