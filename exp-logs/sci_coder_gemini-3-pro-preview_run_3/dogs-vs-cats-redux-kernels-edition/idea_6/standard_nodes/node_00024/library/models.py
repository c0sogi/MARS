import timm
import torch.nn as nn
from library.config import Config


def build_model(model_name: str, pretrained: bool = True):
    """
    Constructs a model using the timm library with the specified backbone.

    The function initializes the model with pre-trained ImageNet weights (if requested)
    and modifies the classification head to output a single logit for binary classification,
    as defined by Config.NUM_CLASSES.

    Args:
        model_name (str): The name of the timm model to create (e.g., 'resnet50.a1_in1k').
        pretrained (bool): Whether to load pre-trained weights. Defaults to True.

    Returns:
        nn.Module: The instantiated PyTorch model.
    """
    # Create the model using timm
    # num_classes=Config.NUM_CLASSES (1) ensures the final layer is replaced
    # with a Linear(in_features, 1) layer, outputting raw logits.
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=Config.NUM_CLASSES
    )

    return model
