import torch
import torch.nn as nn
import timm
from library.config import Config


class EarlyFusionEfficientNet(nn.Module):
    """
    EfficientNet-B2 architecture adapted for Early Fusion of mammography images
    and spatially broadcasted metadata.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=True,
        in_chans=Config.INPUT_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load (e.g., 'efficientnet_b2').
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_chans (int): Number of input channels (3: Image, Age, Implant).
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super().__init__()

        # Backbone
        # EfficientNet-B2 provides a good balance of parameter count (receptive field)
        # and computational efficiency.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Pass through the backbone CNN
        logits = self.backbone(x)

        return logits
