import torch
import torch.nn as nn
import timm
from library.config import (
    INPUT_CHANNELS,
    BACKBONE_NAME,
    DROP_PATH_RATE,
)


class SSBHDNetwork(nn.Module):
    """
    Standardized 2.5D Volumetric Network.

    Directly stacks MRI slices into the channel dimension (64 channels) and feeds them
    into an EfficientNet backbone. This avoids custom adapters which can destabilize training.
    """

    def __init__(self):
        super(SSBHDNetwork, self).__init__()

        # Backbone
        # EfficientNet-B0 configured to accept the 64-channel input directly.
        # We rely on timm's native weight adaptation for the first layer.
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=True,
            in_chans=INPUT_CHANNELS,
            num_classes=1,
            drop_path_rate=DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 64, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Pass through the Backbone
        # The timm model handles Global Average Pooling and the final Linear layer.
        # Shape: (B, 64, 224, 224) -> (B, 1)
        logits = self.backbone(x)

        return logits
