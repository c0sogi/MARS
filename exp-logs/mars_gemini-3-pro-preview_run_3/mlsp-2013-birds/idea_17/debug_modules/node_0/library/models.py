import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    A wrapper class for various CNN backbones using timm.
    Automatically handles the modification of the final classification layer
    to match the number of bird species.
    """

    def __init__(self, backbone_name, pretrained=True):
        """
        Args:
            backbone_name (str): The name of the timm model to create.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super(BirdModel, self).__init__()

        # Create the model using timm
        # num_classes sets the output dimension of the final fully connected layer.
        # in_chans=3 ensures the first layer expects RGB images (replicated spectrograms).
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            in_chans=Config.IN_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw logits of shape (B, NUM_CLASSES).
        """
        return self.backbone(x)


def get_model(model_name, pretrained=True):
    """
    Factory function to instantiate a model from the heterogeneous ensemble list.

    Args:
        model_name (str): Name of the architecture (e.g., 'resnet18', 'efficientnet_b0', 'densenet121').
        pretrained (bool): Whether to initialize with pre-trained weights. Defaults to True.

    Returns:
        nn.Module: The instantiated PyTorch model.

    Raises:
        ValueError: If the provided model_name is not supported in Config.
    """
    if model_name not in Config.MODEL_BACKBONES:
        raise ValueError(
            f"Model '{model_name}' is not supported. "
            f"Available models: {Config.MODEL_BACKBONES}"
        )

    model = BirdModel(model_name, pretrained=pretrained)
    return model
