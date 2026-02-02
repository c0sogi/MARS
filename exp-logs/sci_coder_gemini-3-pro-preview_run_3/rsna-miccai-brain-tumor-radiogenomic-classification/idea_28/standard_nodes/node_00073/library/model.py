import torch
import torch.nn as nn
import timm
from library.data import TOTAL_CHANNELS


class RFMHDNetwork(nn.Module):
    """
    Robust Filename-Sorted Modality-Normalized Network (RFM).

    Architecture:
    - Backbone: EfficientNet-B0 (via timm) configured to accept 64 input channels directly.
      (16 slices * 4 modalities = 64 channels).
    - Uses 'in_chans' argument for native weight adaptation.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load pretrained weights for the backbone.
        """
        super(RFMHDNetwork, self).__init__()

        # Backbone
        # EfficientNet-B0 configured to accept TOTAL_CHANNELS (64).
        # drop_path_rate=0.2 is used for regularization (Stochastic Depth).
        # num_classes=1 creates the linear head for binary classification.
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=TOTAL_CHANNELS,
            drop_path_rate=0.2,
            num_classes=1,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 64, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
