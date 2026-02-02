import torch
import torch.nn as nn
import timm
from library.config import ModelConfig


def create_model(cfg: ModelConfig, pretrained: bool = True) -> nn.Module:
    """
    Instantiates a neural network model based on the provided configuration using timm.

    Args:
        cfg (ModelConfig): Configuration object containing the model architecture name.
        pretrained (bool): If True, loads the pretrained weights specified in the config.

    Returns:
        nn.Module: The PyTorch model with the classifier head replaced by a
                   single-output linear layer (logits) for binary classification.
    """
    # Create the model using timm
    # num_classes=1 configures the head for binary classification (outputting a single logit)
    # in_chans=3 ensures the input layer expects standard RGB images
    model = timm.create_model(
        cfg.model_name, pretrained=pretrained, num_classes=1, in_chans=3
    )

    return model
