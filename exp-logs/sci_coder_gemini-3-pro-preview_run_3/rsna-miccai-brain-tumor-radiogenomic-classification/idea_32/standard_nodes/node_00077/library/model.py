import torch
import torch.nn as nn
import timm
from library.config import TOTAL_CHANNELS


class HRLNNet(nn.Module):
    """
    High-Resolution Layer-Normalized 2.5D Network.
    Backbone: ConvNeXt-Tiny (LayerNorm based).
    Input: (B, 64, 320, 320).
    """

    def __init__(self):
        super(HRLNNet, self).__init__()
        # Load ConvNeXt Tiny
        # in_chans=64 triggers timm to adapt the first conv layer weights
        # drop_path_rate=0.2 for regularization
        # num_classes=0 removes the default head, returning pooled features
        self.backbone = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            in_chans=TOTAL_CHANNELS,
            num_classes=0,
            drop_path_rate=0.2,
        )

        # Get feature dimension (usually 768 for tiny)
        self.num_features = self.backbone.num_features

        # Classification Head
        # Global Average Pooling is handled by the backbone when num_classes=0
        # We add LayerNorm and Linear as per the HRLN-Net design
        self.head = nn.Sequential(
            nn.LayerNorm(self.num_features), nn.Linear(self.num_features, 1)
        )

    def forward(self, x):
        # Input x shape: (B, 64, 320, 320)

        # Extract features using the backbone
        # Output shape: (B, num_features)
        features = self.backbone(x)

        # Pass through the classification head
        # Output shape: (B, 1)
        logits = self.head(features)

        # Squeeze to shape (B,) for compatibility with BCEWithLogitsLoss
        return logits.squeeze(1)
