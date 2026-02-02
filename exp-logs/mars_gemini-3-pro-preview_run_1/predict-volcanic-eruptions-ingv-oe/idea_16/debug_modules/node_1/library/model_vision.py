import torch
import torch.nn as nn
import timm
from library.config import Config


class VolcanoEfficientNet(nn.Module):
    """
    EfficientNet-B0 model modified for multi-channel seismic spectrogram regression.

    This class wraps a timm EfficientNet-B0 architecture, adapting it to handle
    10-channel inputs (representing 10 seismic sensors) and outputting a single
    regression value (log-time-to-eruption).
    """

    def __init__(self, pretrained=True):
        """
        Initialize the VolcanoEfficientNet model.

        Args:
            pretrained (bool): If True, loads ImageNet pretrained weights.
                               The first layer weights are adapted for 10 channels.
        """
        super(VolcanoEfficientNet, self).__init__()

        # Retrieve sensor count from configuration (should be 10)
        self.n_channels = Config.N_SENSORS

        # Create the EfficientNet-B0 model using timm
        # in_chans=10: Adapts the first convolutional layer (conv_stem) to accept 10 input channels.
        #              timm handles weight initialization for the extra channels (e.g., by averaging).
        # num_classes=1: Replaces the final classifier with a linear layer outputting a single value.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=self.n_channels,
            num_classes=1,
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 10, H, W).
                              Expected H, W is 224 based on Config.IMG_SIZE.

        Returns:
            torch.Tensor: Regression output of shape (Batch, 1).
        """
        output = self.backbone(x)
        return output
