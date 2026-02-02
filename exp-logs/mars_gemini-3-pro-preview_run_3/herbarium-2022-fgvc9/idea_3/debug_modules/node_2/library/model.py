import torch
import torch.nn as nn
import timm
from library.config import Config


class PlantConvNeXt(nn.Module):
    """
    PlantConvNeXt model implementation using ConvNeXt-Tiny backbone.

    This class utilizes `timm.create_model` to initialize the 'convnext_tiny'
    architecture. It replaces the classification head to output logits for
    the specific number of plant species (15,501).
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        drop_path_rate=Config.DROP_PATH_RATE,
    ):
        """
        Initialize the PlantConvNeXt model.

        Args:
            model_name (str): Name of the model architecture in timm (default: 'convnext_tiny').
            num_classes (int): Number of output classes (default: 15501).
            pretrained (bool): Whether to load pretrained ImageNet weights (default: True).
            drop_path_rate (float): Stochastic depth rate (default: 0.1).
        """
        super(PlantConvNeXt, self).__init__()

        # Create the model using timm
        # Passing num_classes automatically resets the classifier head to the target size.
        # This projects the 768-dim features to num_classes logits.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.model(x)
