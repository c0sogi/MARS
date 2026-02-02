import timm
import torch
import torch.nn as nn
from library.config import Config


def get_model(
    model_name: str = Config.MODEL_NAME,
    num_classes: int = Config.NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    """
    Initializes the model architecture using timm.

    Args:
        model_name (str): The name of the model architecture in timm.
        num_classes (int): The number of output classes.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        nn.Module: The PyTorch model with the classifier head adapted to num_classes.
    """
    # Create the model using timm
    # num_classes argument automatically replaces the head with a new Linear layer
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes the backbone of the model, leaving only the classification head trainable.
    This is used for the 'Warm-up' phase of the Two-Phase Transfer Learning strategy.

    Args:
        model (nn.Module): The model to modify.
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Identify and unfreeze the classifier head
    # timm models provide a standard interface to access the classifier
    # For ConvNeXt, the attribute is usually 'head'
    if hasattr(model, "get_classifier"):
        classifier = model.get_classifier()
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True
    elif hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    elif hasattr(model, "fc"):
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        # Fallback: try to unfreeze the last module if specific name not found
        # This is a safety catch, though ConvNeXt uses 'head'
        params = list(model.parameters())
        if len(params) > 0:
            params[-1].requires_grad = True
            if len(params) > 1:
                params[-2].requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreezes all parameters in the model to allow full fine-tuning.
    This is used for the second phase of training.

    Args:
        model (nn.Module): The model to modify.
    """
    for param in model.parameters():
        param.requires_grad = True
