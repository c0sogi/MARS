import torch
import torch.nn as nn
import timm
from library.config import CFG


class CatDogModel(nn.Module):
    """
    Wrapper class for timm models to ensure consistent interface and
    handling of binary classification head.
    """

    def __init__(self, model_name, pretrained=True):
        super(CatDogModel, self).__init__()
        # Create the model using timm
        # num_classes=1 sets the final layer to output a single logit
        # in_chans=3 ensures it expects RGB images
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=1, in_chans=3
        )

    def forward(self, x):
        # Forward pass through the timm model
        # Returns logits of shape (batch_size, 1)
        output = self.model(x)
        return output


def get_model(model_name, pretrained=True):
    """
    Factory function to instantiate the model.

    Args:
        model_name (str): The name of the model architecture (e.g., 'tf_efficientnetv2_m.in21k_ft_in1k').
        pretrained (bool): Whether to load ImageNet pre-trained weights.

    Returns:
        nn.Module: The instantiated model ready for training/inference.
    """
    model = CatDogModel(model_name, pretrained=pretrained)
    return model
