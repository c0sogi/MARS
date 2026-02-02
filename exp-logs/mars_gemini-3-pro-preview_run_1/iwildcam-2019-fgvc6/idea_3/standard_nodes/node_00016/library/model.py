import torch
import torch.nn as nn
import timm
import os
from library.config import Config


def create_model(
    model_name=Config.MODEL_NAME,
    num_classes=Config.NUM_CLASSES,
    pretrained=Config.PRETRAINED,
    drop_path_rate=Config.DROP_PATH_RATE,
    checkpoint_path=None,
):
    """
    Creates the animal classification model based on the ConvNeXt architecture.

    Args:
        model_name (str): Name of the timm model to create (default: from Config).
        num_classes (int): Number of output classes (default: from Config).
        pretrained (bool): Whether to load ImageNet pretrained weights (default: from Config).
        drop_path_rate (float): Stochastic depth rate (default: from Config).
        checkpoint_path (str, optional): Path to a local .pth file to load weights from.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    # Create the model using timm
    # timm automatically resets the classifier head if num_classes differs from the pretrained model
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )

    # Load specific checkpoint weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load to CPU first to avoid potential CUDA OOM during initialization
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Extract state_dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Clean state_dict keys (remove 'module.' prefix if trained with DataParallel)
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k[7:]] = v
            else:
                clean_state_dict[k] = v

        # Load weights into model
        # strict=True ensures the architecture exactly matches the weights
        msg = model.load_state_dict(clean_state_dict, strict=True)
        # print(f"Loaded weights from {checkpoint_path}: {msg}")

    return model
