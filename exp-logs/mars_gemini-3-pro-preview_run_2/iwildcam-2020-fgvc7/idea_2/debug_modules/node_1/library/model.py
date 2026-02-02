import torch
import torch.nn as nn
import timm
from library.config import Config


def get_model(pretrained: bool = True) -> nn.Module:
    """
    Initializes and returns the EfficientNetV2-S model.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    print(f"Creating model: {Config.MODEL_NAME}")
    print(f"  Num Classes: {Config.NUM_CLASSES}")
    print(f"  Drop Path Rate: {Config.DROP_PATH_RATE}")
    print(f"  Dropout Rate: {Config.DROPOUT_RATE}")

    # Create the model using timm
    # This handles loading pretrained weights, setting up the backbone,
    # configuring stochastic depth (drop_path_rate), and replacing the head (num_classes).
    model = timm.create_model(
        Config.MODEL_NAME,
        pretrained=pretrained,
        num_classes=Config.NUM_CLASSES,
        drop_path_rate=Config.DROP_PATH_RATE,
        drop_rate=Config.DROPOUT_RATE,
    )

    # Move model to the configured device (GPU/CPU)
    device = torch.device(Config.DEVICE)
    model.to(device)

    return model
