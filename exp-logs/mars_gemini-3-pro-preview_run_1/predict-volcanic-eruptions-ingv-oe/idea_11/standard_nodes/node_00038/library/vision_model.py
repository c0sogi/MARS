import torch
import torch.nn as nn
import timm
from library.config import MODEL_NAME, PRETRAINED, IN_CHANNELS, SEED

# Set fixed seed for reproducibility
torch.manual_seed(SEED)


class VolcanoEfficientNet(nn.Module):
    """
    VolcanoEfficientNet: A Convolutional Neural Network based on EfficientNet-B0,
    adapted for multi-channel seismic spectrogram regression.

    This model modifies the standard EfficientNet architecture to:
    1. Accept 10 input channels (one per seismic sensor) instead of the standard 3 (RGB).
    2. Output a single scalar value (regression) instead of class probabilities.
    """

    def __init__(self):
        """
        Initializes the VolcanoEfficientNet model.

        Uses `timm` to create the backbone. The `in_chans` argument handles the
        structural modification of the first layer, and `num_classes=1` sets up
        the final linear layer for regression.
        """
        super(VolcanoEfficientNet, self).__init__()

        self.backbone = timm.create_model(
            MODEL_NAME, pretrained=PRETRAINED, in_chans=IN_CHANNELS, num_classes=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 10, Height, Width).
                              Represents the global-max-scaled log-mel spectrograms.

        Returns:
            torch.Tensor: Output prediction of shape (Batch, 1).
                          Represents the predicted target (log-transformed time_to_eruption).
        """
        return self.backbone(x)
