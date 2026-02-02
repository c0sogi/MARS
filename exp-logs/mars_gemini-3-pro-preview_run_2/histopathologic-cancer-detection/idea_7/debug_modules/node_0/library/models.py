import timm
import torch.nn as nn
from library.config import Config


def get_model(model_name, pretrained=True, num_classes=1):
    """
    Instantiates a model architecture using the timm library.

    This function creates models with Global Average Pooling (GAP) and a
    classification head adapted for the specified number of classes (default 1).
    It ensures that architecture-specific normalization layers (like LayerNorm
    in ConvNeXt) are preserved in the head.

    Args:
        model_name (str): The name of the architecture to create (e.g., 'convnext_tiny', 'tf_efficientnetv2_s').
        pretrained (bool): If True, loads weights pretrained on ImageNet.
        num_classes (int): Number of output classes. Defaults to 1 for binary classification.

    Returns:
        nn.Module: The PyTorch model.
    """
    # Create the model using timm
    # global_pool='avg' enforces the use of Global Average Pooling as per the task requirements.
    # num_classes adjusts the final linear layer while maintaining the preceding head structure
    # (e.g., normalization layers).
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, global_pool="avg"
    )

    return model
