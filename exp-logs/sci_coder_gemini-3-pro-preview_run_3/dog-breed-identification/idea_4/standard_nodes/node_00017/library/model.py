import torch
import torch.nn as nn
import timm
from library.config import Config


def create_model(num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Creates the ConvNeXt Base model using timm.

    Args:
        num_classes (int): Number of target classes.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        torch.nn.Module: The configured model.
    """
    # Initialize the model using timm
    # This handles downloading weights and replacing the head with a Linear layer for num_classes
    model = timm.create_model(
        Config.MODEL_NAME, pretrained=pretrained, num_classes=num_classes
    )

    return model


def freeze_backbone(model):
    """
    Freezes the backbone of the model, leaving only the classification head trainable.
    Used for the initial warm-up phase.

    Args:
        model (torch.nn.Module): The model to freeze.
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Retrieve the classifier head
    # timm models consistently expose this via get_classifier() or the 'head' attribute
    classifier = model.get_classifier()

    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "head"):
        for param in model.head.parameters():
            param.requires_grad = True
    else:
        # Fallback for generic fc layers if specific attributes aren't found
        # (Unlikely for ConvNeXt, but good for robustness)
        if hasattr(model, "fc"):
            for param in model.fc.parameters():
                param.requires_grad = True


def unfreeze_backbone(model):
    """
    Unfreezes all parameters in the model.
    Used for the fine-tuning phases.

    Args:
        model (torch.nn.Module): The model to unfreeze.
    """
    for param in model.parameters():
        param.requires_grad = True
