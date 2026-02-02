import torch
import torch.nn as nn
import timm
from library.config import Config


class INatModel(nn.Module):
    """
    INatModel encapsulates the neural network architecture for the iNaturalist 2019 task.
    It utilizes an EfficientNetV2-S backbone from the timm library.
    """

    def __init__(self, pretrained: bool = Config.PRETRAINED):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load pre-trained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(INatModel, self).__init__()

        # Create the model using timm
        # passing num_classes automatically replaces the head with a new Linear layer
        # initialized for the specific number of classes.
        self.model = timm.create_model(
            model_name=Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES).
        """
        return self.model(x)
