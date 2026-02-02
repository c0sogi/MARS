import torch
import torch.nn as nn
import timm
from library.config import Config


class PlantClassifier(nn.Module):
    """
    PlantClassifier model based on EfficientNet-B0 architecture.

    This class wraps a timm-based EfficientNet model, ensuring the final
    classification head is adapted to the specific number of plant species
    in the dataset.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
    ):
        """
        Initialize the PlantClassifier.

        Args:
            model_name (str): The name of the architecture to use (default: efficientnet_b0).
            num_classes (int): The number of output classes (default: 32093).
            pretrained (bool): Whether to initialize with pretrained weights (default: True).
        """
        super(PlantClassifier, self).__init__()

        # Create the model using timm.
        # This function handles:
        # 1. Loading the architecture configuration.
        # 2. Loading pretrained weights (if pretrained=True).
        # 3. Replacing the original ImageNet classifier head with a new Linear layer
        #    with 'num_classes' outputs.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Perform the forward pass.

        Args:
            x (torch.Tensor): Input batch of images with shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Raw logits with shape (Batch, Num_Classes).
        """
        return self.model(x)
