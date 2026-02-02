import torch
import torch.nn as nn
import timm
from library.config import Config


class SHDNet(nn.Module):
    """
    Stabilized High-Density 2.5D Network (SHD-Net).

    This architecture is designed to ingest high-density volumetric MRI data (128 channels)
    by first compressing it through a stabilized stem before passing it to a standard
    EfficientNet-B0 backbone.
    """

    def __init__(self, drop_path_rate=None):
        """
        Args:
            drop_path_rate (float, optional): Stochastic depth rate.
                                              Defaults to Config.DROP_PATH_RATE.
        """
        super(SHDNet, self).__init__()

        # Use config default if not provided
        if drop_path_rate is None:
            drop_path_rate = Config.DROP_PATH_RATE

        # Stabilized Compression Stem
        # Compresses 128 channels (32 slices * 4 mods) to 64 channels
        # This bridges the gap between high-density input and the backbone
        self.stem = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Initialize Stem with Kaiming Normal
        # Critical for stability when projecting high-dim input to lower-dim features
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # Backbone: EfficientNet-B0
        # in_chans=64 to match stem output
        # num_classes=0 to return feature maps instead of logits
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=64,
            drop_path_rate=drop_path_rate,
            num_classes=0,
        )

        # Head
        # Global Average Pooling followed by a single fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 256, 256)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Stabilized Compression
        # x: (B, 128, 256, 256) -> (B, 64, 256, 256)
        x = self.stem(x)

        # 2. Backbone Feature Extraction
        # x: (B, 64, 256, 256) -> (B, C, H, W)
        x = self.backbone.forward_features(x)

        # 3. Classification Head
        # x: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        x = self.global_pool(x).flatten(1)

        # x: (B, C) -> (B, 1)
        x = self.fc(x)

        return x
