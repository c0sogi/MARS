import torch
import torch.nn as nn
import timm
from library.config import Config


class PathologyModel(nn.Module):
    """
    PathologyModel based on the ConvNeXt-Tiny architecture.

    This model implements the specific head design requirements:
    - Global Average Pooling (GAP)
    - Native LayerNorm before the classifier
    - Single fully connected layer
    - Exclusion of Multi-Sample Dropout (drop_rate=0.0)
    - Inclusion of Stochastic Depth (DropPath) in the backbone
    """

    def __init__(self, model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED):
        super(PathologyModel, self).__init__()

        # Initialize the ConvNeXt model using timm
        # num_classes=1: Sets up the final linear layer for binary classification
        # drop_rate=0.0: Ensures no dropout is applied in the head (excludes MSD)
        # drop_path_rate: Applies stochastic depth regularization to the backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.HEAD_DROPOUT,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, 1).
        """
        return self.backbone(x)
