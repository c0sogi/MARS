import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """
    Initializes the ConvNeXt-Tiny model.

    Args:
        num_classes (int): Number of output classes (dog breeds).
        pretrained (bool): Whether to load ImageNet-1k pre-trained weights.

    Returns:
        torch.nn.Module: The configured model.
    """
    # Determine weights
    weights = "IMAGENET1K_V1" if pretrained else None

    # Load model
    # Dynamically load model based on Config.MODEL_NAME (e.g., convnext_small)
    model = getattr(models, Config.MODEL_NAME)(weights=weights)

    # The torchvision ConvNeXt classifier is a Sequential block:
    # (0): LayerNorm2d((768,), eps=1e-06, elementwise_affine=True)
    # (1): Flatten(start_dim=1, end_dim=-1)
    # (2): Linear(in_features=768, out_features=1000, bias=True)

    # We want to replace the final Linear layer to match our number of classes.
    # We access the input features of the existing linear layer to ensure compatibility.
    last_layer_idx = len(model.classifier) - 1
    in_features = model.classifier[last_layer_idx].in_features

    # Replace the last layer
    model.classifier[last_layer_idx] = nn.Linear(in_features, num_classes)

    return model


def freeze_backbone(model):
    """
    Freezes the backbone (features) of the model and ensures the classifier is trainable.
    Used for the warm-up phase.

    Args:
        model (torch.nn.Module): The ConvNeXt model.
    """
    # Freeze the feature extractor backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Ensure the classifier head is trainable
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """
    Unfreezes all parameters in the model.
    Used for the fine-tuning phase.

    Args:
        model (torch.nn.Module): The ConvNeXt model.
    """
    for param in model.parameters():
        param.requires_grad = True
