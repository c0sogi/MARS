import torch
import torch.nn as nn
import timm
from library.config import Config


class S3HDNetwork(nn.Module):
    """
    Semantically-Structured Stabilized High-Density (S3HD) Network.

    Architecture:
    1. Input: (B, 128, 224, 224) - 32 slices x 4 modalities.
    2. Stabilized Global-Mixing Stem: Compresses 128ch -> 64ch with He Init.
    3. Backbone: EfficientNet-B0 (in_chans=64, drop_path=0.2).
    4. Head: Global Average Pooling + FC (Logits).
    """

    def __init__(self):
        super(S3HDNetwork, self).__init__()

        # Input channels from Config (expected 128)
        input_channels = Config.INPUT_CHANNELS
        stem_channels = 64

        # 1. Stabilized Global-Mixing Stem
        # Performs Compression, Early Fusion, and Downsampling
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Explicit Initialization for Stability
        # Critical for preventing gradient explosion with high-channel inputs
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # 2. Backbone: EfficientNet-B0
        # Instantiated via timm with specific configuration
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,  # efficientnet_b0
            pretrained=True,
            in_chans=stem_channels,  # 64, matches stem output
            num_classes=Config.NUM_CLASSES,  # 1
            drop_path_rate=Config.DROP_PATH_RATE,  # 0.2
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
                          Note: Sigmoid is not applied here to allow
                          usage of BCEWithLogitsLoss for numerical stability.
        """
        # Pass through Stabilized Stem
        # Shape: (B, 128, 224, 224) -> (B, 64, 112, 112)
        x = self.stem(x)

        # Pass through Backbone
        # Shape: (B, 64, 112, 112) -> (B, 1)
        logits = self.backbone(x)

        return logits
