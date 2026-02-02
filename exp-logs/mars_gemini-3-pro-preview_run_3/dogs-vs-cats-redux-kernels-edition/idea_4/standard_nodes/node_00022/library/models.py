import torch
import torch.nn as nn
import timm
from library.config import Config


def build_model(backbone_name, pretrained=True):
    """
    Constructs the neural network model based on the specified backbone.

    This factory function utilizes the `timm` library to create models with
    pretrained weights (ImageNet). It automatically replaces the final
    classification layer (head) to match the number of classes defined in
    Config (1 for binary classification).

    Args:
        backbone_name (str): Name of the backbone architecture.
                             Supported: 'resnet101', 'convnext_small'.
        pretrained (bool): Whether to initialize with pretrained ImageNet weights.
                           Defaults to True.

    Returns:
        nn.Module: The configured PyTorch model ready for training.
    """

    # Validate backbone name against Config to ensure consistency
    if backbone_name not in Config.MODEL_BACKBONES:
        # While we could support any timm model, restricting to Config ensures
        # the ensemble strategy is followed.
        raise ValueError(
            f"Backbone '{backbone_name}' is not in the supported list: {Config.MODEL_BACKBONES}"
        )

    # Create the model using timm
    # num_classes=Config.NUM_CLASSES (1) ensures the final layer is a linear layer
    # with 1 output unit, suitable for BCEWithLogitsLoss.
    model = timm.create_model(
        backbone_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES, in_chans=3
    )

    return model
