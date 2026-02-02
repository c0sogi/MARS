import torch
import torch.nn as nn
import timm
from library.config import CFG


class CatDogModel(nn.Module):
    """
    Wrapper class for timm models adapted for binary classification.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        """
        Args:
            model_name (str): The name of the model architecture to load from timm.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(CatDogModel, self).__init__()

        # Create the model using timm
        # num_classes=1 sets the final fully connected layer to output a single logit
        # for binary classification (BCEWithLogitsLoss).
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Logits (raw scores) from the model.
        """
        return self.model(x)


def build_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to instantiate a model.

    Args:
        model_name (str): Name of the model architecture (e.g., from CFG.model_names).
        pretrained (bool): Whether to initialize with pretrained weights.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    model = CatDogModel(model_name=model_name, pretrained=pretrained)
    return model
