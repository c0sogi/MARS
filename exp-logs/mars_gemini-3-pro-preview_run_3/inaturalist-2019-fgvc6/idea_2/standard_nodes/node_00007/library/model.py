import torch
import timm
from library import config


def create_model(num_classes=config.NUM_CLASSES, pretrained=True):
    """
    Creates the EfficientNetV2-M model using timm.

    Args:
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        torch.nn.Module: The configured model.
    """
    model = timm.create_model(
        config.MODEL_NAME, pretrained=pretrained, num_classes=num_classes
    )
    return model


def set_backbone_trainable(model, trainable):
    """
    Sets the backbone parameters to be trainable or frozen.

    Args:
        model (torch.nn.Module): The model to modify.
        trainable (bool): If True, all parameters are trainable.
                          If False, backbone is frozen and only the classifier head is trainable.
    """
    if trainable:
        # Unfreeze all parameters for full fine-tuning (Stage 1 & 2)
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only the classifier head (Stage 3)
        # timm models provide a standard interface to access the classifier
        classifier = model.get_classifier()
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True
        else:
            # Fallback: Attempt to identify classifier by common names if get_classifier fails
            # (Though get_classifier should work for EfficientNetV2 in timm)
            head_found = False
            for name in ["classifier", "head", "fc"]:
                if hasattr(model, name):
                    module = getattr(model, name)
                    for param in module.parameters():
                        param.requires_grad = True
                    head_found = True
                    break

            if not head_found:
                # If we can't find the head, we warn by printing (or just pass silently as per instructions)
                pass
