import torch
import torch.nn as nn
import timm
from library.data import TOTAL_CHANNELS


class RFMHDNetwork(nn.Module):
    """
    Robust Filename-Sorted Modality-Normalized High-Density (RFM-HD) Network.

    Architecture:
    1. Stabilized Global-Mixing Stem: Compresses high-density volumetric input (128 channels)
       to a stable feature space (64 channels) using a standard convolution to mix
       all modalities and slices immediately.
    2. Backbone: EfficientNet-B0 (via timm) configured to accept 64 channels.
    3. Head: Global Average Pooling + Linear Layer (handled by timm).
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super(RFMHDNetwork, self).__init__()

        # 1. Stabilized Global-Mixing Stem
        # Input: (B, 128, 224, 224) -> Output: (B, 64, 112, 112)
        # We use stride=2 to downsample spatial resolution early, matching EfficientNet's typical start.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=TOTAL_CHANNELS,  # 128
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 2. Backbone
        # EfficientNet-B0 configured to accept the 64-channel output from the stem.
        # drop_path_rate=0.2 is used for regularization (Stochastic Depth).
        # num_classes=1 creates the linear head for binary classification.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=64,
            drop_path_rate=0.2,
            num_classes=1,
        )

        # Explicitly initialize the stem to ensure stability with high-channel inputs
        self._init_weights()

    def _init_weights(self):
        """
        Applies Kaiming/He Normal initialization to the Stem layers.
        This prevents gradient explosion when projecting 128 channels to 64.
        """
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
                          (Sigmoid should be applied externally or via BCEWithLogitsLoss).
        """
        # 1. Pass through Stabilized Stem
        x = self.stem(x)

        # 2. Pass through Backbone (includes GAP and Classifier Head)
        x = self.backbone(x)

        return x
