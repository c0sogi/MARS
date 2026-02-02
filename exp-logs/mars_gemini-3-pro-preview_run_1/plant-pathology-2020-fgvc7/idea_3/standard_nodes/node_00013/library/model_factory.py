import torch.nn as nn
import timm
from library.config import Config


def get_model(config: Config) -> nn.Module:
    """
    Constructs the neural network architecture.
    Initializes the ResNeXt-50 32x4d backbone with pretrained weights
    and replaces the head for the specific number of classes.

    Args:
        config (Config): Configuration object containing model settings.

    Returns:
        nn.Module: The PyTorch model ready for training.
    """
    # Create the model using timm library
    # model_name is expected to be 'resnext50_32x4d' as per config
    # pretrained=True loads ImageNet weights
    # num_classes=4 replaces the final FC layer automatically to match the target columns
    model = timm.create_model(
        config.model_name, pretrained=config.pretrained, num_classes=config.num_classes
    )

    return model
