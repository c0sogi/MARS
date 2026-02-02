import torch
import torch.nn as nn
import timm
from library.config import Config


class WITSNet(nn.Module):
    """
    Weight-Inited Thick-Slab Independent Instance Network (WITS-Net).

    Architecture:
        - Backbone: EfficientNet-B0 (pretrained on ImageNet).
        - Input: 9 Channels (3x FLAIR, 3x T1wCE, 3x T2w).
        - Output: Binary Logits (for MGMT promoter methylation prediction).

    Structural Innovation:
        - Weight Inflation Initialization: Adapts pretrained RGB weights to 9-channel input
          by distributing channel energy, allowing immediate processing of volumetric slabs
          without training from scratch or using learnable adapters.
    """

    def __init__(self):
        super(WITSNet, self).__init__()

        # Initialize the EfficientNet-B0 backbone
        # We use the standard 3-channel input to preserve pretrained priors
        # (Cite solution_lesson_node_00009)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.DROPOUT_RATE,
            in_chans=Config.IN_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, Height, Width).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        return self.backbone(x)
