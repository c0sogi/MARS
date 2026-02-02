import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(pretrained: bool = True) -> nn.Module:
    """
    Instantiates the ConvNeXt-Small model using the timm library.

    The model is configured with the number of classes specified in Config.NUM_CLASSES.

    Args:
        pretrained (bool): Whether to load pretrained ImageNet-22k weights.
                           Defaults to True.

    Returns:
        nn.Module: The PyTorch model ready for training or inference.
    """
    # Create the model using timm
    # num_classes ensures the final head is replaced to match our dataset (120 breeds)
    model = timm.create_model(
        Config.MODEL_NAME, pretrained=pretrained, num_classes=Config.NUM_CLASSES
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    """
    Freezes the backbone parameters of the model, leaving only the classification head trainable.

    This is used during Phase 1 (Linear Probing) of the training strategy to align the
    randomly initialized head with the pretrained features without destroying them.

    Args:
        model (nn.Module): The model instance to modify.
    """
    # 1. Freeze all parameters in the network
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze the classification head
    # ConvNeXt architectures in timm use the attribute 'head' for the final classifier.
    # We check for common classifier names to ensure robustness if the architecture changes.
    head_unfrozen = False

    # Check for 'head' (ConvNeXt, ViT, Swin)
    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        for param in model.head.parameters():
            param.requires_grad = True
        head_unfrozen = True

    # Check for 'fc' (ResNet)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        for param in model.fc.parameters():
            param.requires_grad = True
        head_unfrozen = True

    # Check for 'classifier' (EfficientNet, MobileNet)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        for param in model.classifier.parameters():
            param.requires_grad = True
        head_unfrozen = True

    if not head_unfrozen:
        # Fallback: Identify the last linear layer if named attributes fail
        # This is a safety net, though 'head' should exist for ConvNeXt.
        print(
            "Warning: Could not identify standard head attributes ('head', 'fc', 'classifier')."
        )
        print("Attempting to unfreeze the last module parameters manually.")
        params = list(model.parameters())
        if len(params) > 0:
            # Unfreeze the last layer's weights and bias
            params[-1].requires_grad = True  # Bias
            if len(params) > 1:
                params[-2].requires_grad = True  # Weights


def unfreeze_backbone(model: nn.Module) -> None:
    """
    Unfreezes all parameters in the model.

    This is used during Phase 2 (Fine-Tuning) of the training strategy to allow
    the entire network to adapt to the specific dog breed features.

    Args:
        model (nn.Module): The model instance to modify.
    """
    for param in model.parameters():
        param.requires_grad = True
