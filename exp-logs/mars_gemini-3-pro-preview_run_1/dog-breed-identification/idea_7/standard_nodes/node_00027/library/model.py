import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(model_config):
    """
    Creates and returns a model based on the provided configuration.

    Args:
        model_config (dict): Dictionary containing model parameters like:
                             - model_name (str)
                             - pretrained (bool)
                             - num_classes (int)
                             - drop_rate (float)
                             - drop_path_rate (float)

    Returns:
        model (nn.Module): The initialized PyTorch model.
    """
    model_name = model_config["model_name"]
    pretrained = model_config["pretrained"]
    num_classes = model_config["num_classes"]
    drop_rate = model_config.get("drop_rate", 0.0)
    drop_path_rate = model_config.get("drop_path_rate", 0.0)

    # Create the model using timm
    # timm handles the replacement of the classification head when num_classes is specified
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    return model


def set_backbone_trainable(model, trainable=True):
    """
    Freezes or unfreezes the backbone parameters of the model.
    Used for the Head Adaptation phase where we only want to train the classifier.

    Args:
        model (nn.Module): The model to modify.
        trainable (bool): If True, all parameters are trainable.
                          If False, backbone is frozen, only head is trainable.
    """
    if trainable:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head
        # ConvNeXt and Swin typically use 'head' in timm.
        # We check for common classifier attribute names to be robust.
        classifier_layer = None
        if hasattr(model, "head"):
            classifier_layer = model.head
        elif hasattr(model, "fc"):
            classifier_layer = model.fc
        elif hasattr(model, "classifier"):
            classifier_layer = model.classifier

        if classifier_layer is not None:
            for param in classifier_layer.parameters():
                param.requires_grad = True
        else:
            # Fallback: If no known head attribute is found, warn or try to find last module.
            # For the specific models in Config (ConvNeXt, Swin), 'head' is the standard attribute.
            print(
                "Warning: Could not identify classifier head to keep trainable. Model might be fully frozen."
            )
