import timm
import torch.nn as nn
from library.config import MODEL_RESNET, MODEL_CONVNEXT, MODEL_MAXVIT


def create_model(model_name, pretrained=True, num_classes=1):
    """
    Initializes the specified architecture using timm, loads pretrained weights,
    and modifies the classification head to output the specified number of classes.

    Args:
        model_name (str): The name of the model architecture (e.g., 'resnet50.a1_in1k').
        pretrained (bool): Whether to initialize with pretrained weights (e.g., ImageNet).
        num_classes (int): Number of output classes. Defaults to 1 for binary classification.

    Returns:
        model (nn.Module): The initialized PyTorch model.
    """
    try:
        # Create the model using timm.
        # This handles:
        # 1. Architecture instantiation.
        # 2. Loading pretrained weights if requested.
        # 3. Re-initializing the head (fc/classifier) for num_classes.
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}'. Error: {e}")

    return model
