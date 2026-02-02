import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(num_classes, device):
    """
    Loads the EfficientNet-B3 architecture with Noisy Student weights via timm.
    Cite solution_lesson_node_00006: Synergistic Model Scaling.

    Args:
        num_classes (int): The number of output classes (target categories).
        device (str or torch.device): The device ('cpu' or 'cuda') to move the model to.

    Returns:
        torch.nn.Module: The modified model ready for training/inference.
    """
    # Load EfficientNet-B3 with Noisy Student weights
    # drop_path_rate=0.2 acts as a regularizer (Stochastic Depth)
    model = timm.create_model(
        Config.MODEL_NAME, pretrained=True, num_classes=num_classes, drop_path_rate=0.2
    )

    # Move the model to the specified computation device
    model = model.to(device)

    return model
