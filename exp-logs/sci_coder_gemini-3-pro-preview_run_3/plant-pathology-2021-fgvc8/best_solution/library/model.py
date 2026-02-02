import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleClassifier(nn.Module):
    """
    Apple Disease Classifier using a ConvNeXt-Tiny backbone.

    Attributes:
        model (nn.Module): The backbone model with a modified classification head.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        drop_path_rate=0.1,
    ):
        """
        Initialize the model.

        Args:
            model_name (str): Name of the model architecture (default: convnext_tiny).
            pretrained (bool): Whether to load pretrained weights (default: True).
            num_classes (int): Number of output classes (default: 6).
            drop_path_rate (float): Rate for Stochastic Depth (DropPath) regularization.
        """
        super(AppleClassifier, self).__init__()

        # Create the ConvNeXt model using timm
        # num_classes ensures the head is replaced to output the correct number of logits
        # drop_path_rate enables Stochastic Depth regularization within the blocks
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Raw logits for each class.
        """
        return self.model(x)
