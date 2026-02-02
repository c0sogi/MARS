import torch
import torch.nn as nn
import timm
from library.config import Config


def build_model(num_classes=Config.num_classes, pretrained=Config.pretrained):
    """
    Builds the ConvNeXt-Base model customized for the dog breed classification task.

    Args:
        num_classes (int): The number of output classes (120 for this dataset).
        pretrained (bool): Whether to load weights pretrained on ImageNet.

    Returns:
        nn.Module: The PyTorch model ready for training.
    """
    print(f"Building model: {Config.model_name}")
    print(f"  - Pretrained: {pretrained}")
    print(f"  - Num Classes: {num_classes}")
    print(f"  - Input Resolution: {Config.img_size}x{Config.img_size}")

    # Instantiate the model using timm
    # passing num_classes replaces the head with a Linear layer of the correct size
    model = timm.create_model(
        Config.model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model
