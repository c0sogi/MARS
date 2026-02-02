import torch
import torch.nn as nn
import timm
from library.config import Config


class AppleMaxViT(nn.Module):
    """
    AppleMaxViT model implementation using the MaxViT-Tiny architecture.

    This class initializes a MaxViT backbone from the timm library, configured
    for the specific image size and number of classes defined in the Config.
    It enforces Global Average Pooling (GAP) and a standard linear classifier head.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load ImageNet-1k pretrained weights.
                               Defaults to True.
        """
        super(AppleMaxViT, self).__init__()

        # Initialize the MaxViT model using timm
        # - model_name: Defined in Config (maxvit_tiny_tf_384.in1k)
        # - pretrained: Loads weights trained on ImageNet-1k
        # - num_classes: Replaces the head with a Linear layer for 6 classes
        # - global_pool: Enforces Global Average Pooling ('avg') to avoid learnable pooling
        self.model = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=Config.NUM_CLASSES,
            global_pool="avg",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES).
        """
        return self.model(x)
