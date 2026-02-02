import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaModel(nn.Module):
    """
    CassavaModel based on EfficientNet-B4 (Noisy Student).
    Cite solution_lesson_node_00007: Scaling Classification Performance via Cardinality (and Capacity).

    Uses timm library to load EfficientNet-B4.
    """

    def __init__(
        self,
        pretrained: bool = Config.PRETRAINED,
        num_classes: int = Config.NUM_CLASSES,
    ):
        """
        Initialize the CassavaModel.

        Args:
            pretrained (bool): If True, loads weights pre-trained on ImageNet.
            num_classes (int): The number of output classes.
        """
        super(CassavaModel, self).__init__()

        # Load EfficientNet-B4 with Noisy Student weights
        # drop_rate and drop_path_rate added for regularization
        self.model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images with shape (B, C, H, W).

        Returns:
            torch.Tensor: Raw output logits with shape (B, num_classes).
        """
        return self.model(x)
