import torch
import torch.nn as nn
import timm
from library.config import (
    INPUT_CHANNELS,
    ADAPTER_OUT_CHANNELS,
    BACKBONE_NAME,
    DROP_PATH_RATE,
)


class SSBHDNetwork(nn.Module):
    """
    Stabilized Semantic-Block High-Density (SSB-HD) Network.

    This architecture is designed to handle high-density volumetric MRI inputs (128 channels)
    by compressing them into a stable feature space using a specialized adapter before
    feeding them into a standard EfficientNet backbone.
    """

    def __init__(self):
        super(SSBHDNetwork, self).__init__()

        # 1. Stabilized High-Density Adapter
        # Function: Compresses 128 input channels (32 slices * 4 modalities) to 64 channels.
        # Uses a standard 3x3 convolution to perform early global channel mixing, allowing
        # the model to learn pixel-level correlations across all slices and modalities immediately.
        self.adapter = nn.Sequential(
            nn.Conv2d(
                in_channels=INPUT_CHANNELS,
                out_channels=ADAPTER_OUT_CHANNELS,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ADAPTER_OUT_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # Initialization: Explicitly initialize the adapter with Kaiming Normal.
        # This projects the high-dimensional input into a statistically stable feature space,
        # preventing gradient explosion/vanishing often seen when feeding high-channel inputs
        # directly to pretrained backbones.
        nn.init.kaiming_normal_(
            self.adapter[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # 2. Backbone
        # EfficientNet-B0 configured to accept the 64-channel output from the adapter.
        # Drop Path (Stochastic Depth) is enabled for regularization.
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=True,
            in_chans=ADAPTER_OUT_CHANNELS,
            num_classes=1,
            drop_path_rate=DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 128, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Pass through the Stabilized Adapter
        # Shape: (B, 128, 224, 224) -> (B, 64, 224, 224)
        x = self.adapter(x)

        # Pass through the Backbone
        # The timm model handles Global Average Pooling and the final Linear layer.
        # Shape: (B, 64, 224, 224) -> (B, 1)
        logits = self.backbone(x)

        return logits
