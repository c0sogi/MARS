import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaClassifier(nn.Module):
    """
    Cassava Leaf Disease Classifier using the ConvNeXt architecture.
    Wraps the timm library to provide a configured model with a custom classification head.
    """

    def __init__(
        self,
        model_name: str = Config.MODEL_NAME,
        pretrained: bool = Config.PRETRAINED,
        num_classes: int = Config.NUM_CLASSES,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the model architecture in timm (e.g., 'convnext_tiny').
            pretrained (bool): Whether to load pre-trained ImageNet weights.
            num_classes (int): Number of output classes (5 for this task).
        """
        super(CassavaClassifier, self).__init__()

        # Create the model using timm
        # passing num_classes tells timm to replace the original head with one suited for our specific number of classes
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input batch of images [Batch, Channels, Height, Width].

        Returns:
            torch.Tensor: Raw logits [Batch, Num_Classes].
        """
        return self.model(x)
