import timm
import torch
import torch.nn as nn
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger(name="model_factory")


def create_model(
    model_name: str, num_classes: int = Config.NUM_CLASSES, pretrained: bool = True
):
    """
    Creates a model using the timm library with a custom classification head.

    Args:
        model_name (str): The name of the architecture to create (e.g., 'convnext_small').
        num_classes (int): The number of output classes for the classification head.
        pretrained (bool): Whether to load pretrained weights (usually ImageNet).

    Returns:
        torch.nn.Module: The instantiated model.
    """
    logger.info(f"Creating model: {model_name}")
    logger.info(f"Pretrained: {pretrained}, Num Classes: {num_classes}")

    try:
        # Create the model using timm
        # timm handles replacing the head with a new linear layer when num_classes is specified
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        # Log the parameter count
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model {model_name} created with {n_parameters:,} trainable parameters."
        )

        return model

    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        raise e


def set_backbone_trainable(model: nn.Module, trainable: bool = True):
    """
    Helper function to freeze or unfreeze the backbone of the model while keeping the
    classifier head trainable. This supports the Two-Phase Transfer Learning strategy.

    Args:
        model (nn.Module): The model to modify.
        trainable (bool): If True, unfreezes the backbone. If False, freezes the backbone.
    """
    # First, set the requires_grad attribute for all parameters based on the 'trainable' flag
    for param in model.parameters():
        param.requires_grad = trainable

    # If we are freezing the backbone (trainable=False), we must ensure the classifier
    # head remains trainable.
    if not trainable:
        # timm models provide a standard interface to get the classifier
        classifier = model.get_classifier()
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True
            logger.info("Backbone frozen. Classifier head remains trainable.")
        else:
            logger.warning(
                "Could not identify classifier head. Entire model is frozen."
            )
    else:
        logger.info("Backbone unfrozen. All parameters are trainable.")
