import timm
import torch
import torch.nn as nn
from library.config import Config


def get_model(pretrained=Config.pretrained, device=Config.device):
    """
    Constructs the ConvNeXt-Small model using timm.

    The function instantiates the model architecture specified in Config.
    It handles modifying the classifier head for binary classification (1 output node)
    by passing num_classes to the factory function.

    Args:
        pretrained (bool): If True, loads weights pre-trained on ImageNet.
        device (str): The device ('cpu' or 'cuda') to move the model to.

    Returns:
        model (nn.Module): The PyTorch model ready for training or inference.
    """
    # Instantiate the model
    # num_classes=1 configures the head for binary classification (outputting raw logits)
    model = timm.create_model(
        Config.model_name, pretrained=pretrained, num_classes=Config.num_classes
    )

    # Move model to the computation device
    if device:
        model = model.to(device)

    return model
