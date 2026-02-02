import torch
import torch.nn as nn
import timm
from library.config import Config


class SETIModel(nn.Module):
    """
    Model for SETI technosignature detection.

    This model utilizes a pretrained backbone (via timm) and modifies
    the input stem to accept 6 channels, corresponding to the 6 cadence positions (ABACAD).
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            model_name (str): Name of the model architecture.
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_channels (int): Number of input channels (6 for SETI cadence).
            num_classes (int): Number of output classes (1 for binary classification).
        """
        super(SETIModel, self).__init__()

        # Create the model using timm.
        # specifying in_chans=in_channels triggers timm's adaptation logic:
        # 1. It replaces the first conv layer with one having 6 input channels.
        # 2. It initializes the weights by recycling/repeating the original 3-channel weights.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input spectrograms of shape (Batch, 6, H, W).

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1).
        """
        return self.backbone(x)
