import timm
import torch
import torch.nn as nn
from library.config import Config


def get_model(cfg: Config):
    """
    Instantiates the Swin Transformer model using timm with the specified configuration.

    The function loads the architecture defined in cfg.model_name (e.g., 'swin_tiny_patch4_window7_224'),
    initializes it with ImageNet-1k pretrained weights, and modifies the classification
    head to match the number of classes in the dataset (32,093). It also applies
    stochastic depth regularization via drop_path_rate.

    Args:
        cfg (Config): Configuration object containing:
            - model_name (str): Name of the model architecture.
            - pretrained (bool): Whether to load pretrained weights.
            - num_classes (int): Number of target classes.
            - drop_path_rate (float): Stochastic depth rate.
            - device (torch.device): Device to move the model to.

    Returns:
        torch.nn.Module: The configured Swin Transformer model on the specified device.
    """

    # Create the model using timm
    # passing num_classes tells timm to replace the original ImageNet head (1000 classes)
    # with a new Linear layer initialized for the specific number of classes in our dataset.
    model = timm.create_model(
        model_name=cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_path_rate=cfg.drop_path_rate,
    )

    # Move the model to the computation device (GPU/CPU) defined in the config
    model.to(cfg.device)

    return model
