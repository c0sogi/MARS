import torch
import torch.nn as nn
import timm
from library.config import Config


class PetModel(nn.Module):
    """
    A wrapper class for creating image classification models using the timm library.
    It automatically configures the model for binary classification (1 output logit).
    """

    def __init__(self, model_name, pretrained=True):
        """
        Args:
            model_name (str): The name of the architecture to load (e.g., 'convnext_small.fb_in22k').
            pretrained (bool): Whether to load pretrained weights. Defaults to True.
        """
        super(PetModel, self).__init__()

        # Create the model using timm
        # num_classes=1 configures the head to output a single logit,
        # which is compatible with BCEWithLogitsLoss.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Raw logits of shape (batch_size, 1).
        """
        # The timm model returns the logits directly
        return self.model(x)


def get_model(model_name, pretrained=True):
    """
    Factory function to instantiate a PetModel.

    Args:
        model_name (str): The name of the architecture.
        pretrained (bool): Whether to load pretrained weights.

    Returns:
        PetModel: The instantiated model.
    """
    model = PetModel(model_name, pretrained=pretrained)
    return model
