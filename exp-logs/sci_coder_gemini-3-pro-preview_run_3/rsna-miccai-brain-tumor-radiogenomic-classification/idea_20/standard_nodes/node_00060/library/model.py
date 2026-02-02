import torch
import torch.nn as nn
import timm
from library.config import Config


class MGSHDNetwork(nn.Module):
    """
    Modality-Grouped Stabilized High-Density (MG-SHD) Network.

    This 2.5D CNN architecture is designed to process high-density volumetric MRI data.
    It ingests a 128-channel input (32 slices x 4 modalities) and uses a specialized
    stem to compress this information into a stable feature space before passing it
    to an EfficientNet-B0 backbone.
    """

    def __init__(self):
        super(MGSHDNetwork, self).__init__()

        # ==========================================
        # 1. Stabilized Compression Stem
        # ==========================================
        # Compresses the high-density input (128 channels) to a standard feature depth (64).
        # This performs Early Fusion of the modality-grouped slices.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=Config.IN_CHANS,
                out_channels=Config.STEM_CHANS,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Config.STEM_CHANS),
            nn.ReLU(inplace=True),
        )

        # Explicit Initialization: Kaiming/He Normal
        # This is critical to prevent gradient explosion when projecting the
        # high-dimensional input (128 channels) into the feature space.
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # ==========================================
        # 2. Backbone & Head
        # ==========================================
        # EfficientNet-B0 from timm.
        # - in_chans=64: Accepts the output of the stem.
        # - drop_path_rate=0.2: Stochastic Depth for regularization.
        # - num_classes=1: Creates a head with GAP + Linear layer outputting a single logit.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            in_chans=Config.STEM_CHANS,
            num_classes=1,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 256, 256).
                              128 channels = 4 modalities * 32 slices.

        Returns:
            torch.Tensor: Output logits of shape (B, 1).
        """
        # 1. Pass through Stabilized Compression Stem
        # Shape: (B, 128, 256, 256) -> (B, 64, 256, 256)
        x = self.stem(x)

        # 2. Pass through Backbone
        # Internally performs feature extraction, Global Average Pooling, and Linear projection.
        # Shape: (B, 64, 256, 256) -> (B, 1)
        x = self.backbone(x)

        return x
