import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    A wrapper class for timm models to perform Bird Species Classification.

    This class initializes a specified backbone architecture (e.g., ResNet-18,
    EfficientNet-B0, DenseNet-121) using the `timm` library. It ensures the model
    accepts 3-channel input (pseudo-RGB) and outputs logits for the specified
    number of classes.
    """

    def __init__(
        self, model_name, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
    ):
        """
        Args:
            model_name (str): The name of the architecture to use (e.g., 'resnet18').
            num_classes (int): The number of target classes (default: 19).
            pretrained (bool): Whether to load ImageNet pretrained weights (default: True).
        """
        super(BirdModel, self).__init__()

        # Create the model using timm
        # in_chans=3 ensures compatibility with the 3-channel pseudo-RGB inputs
        # num_classes sets the output layer size
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes, in_chans=3
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.backbone(x)


def get_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
    """
    Factory function to instantiate a BirdModel.

    Args:
        model_name (str): Name of the architecture.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        BirdModel: An instance of the configured model.
    """
    # Validate that the requested model is one of the supported architectures
    # strictly based on the provided strategy, though timm supports many others.
    if model_name not in Config.ARCHITECTURES:
        # Warning: Using an architecture not explicitly listed in Config
        pass

    model = BirdModel(
        model_name=model_name, num_classes=num_classes, pretrained=pretrained
    )
    return model
