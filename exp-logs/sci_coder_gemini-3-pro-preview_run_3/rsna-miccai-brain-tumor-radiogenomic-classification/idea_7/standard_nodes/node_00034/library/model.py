import torch
import torch.nn as nn
import timm
from library import config


class MGMTNet(nn.Module):
    """
    2.5D Stacked Network using Native Input Adaptation.

    Architecture:
    1. Input: (B, 64, 224, 224) - 64 channels from 4 modalities * 16 slices.
    2. Backbone: EfficientNet-B0 initialized with in_chans=64.
       - Uses timm's native weight recycling to adapt pretrained RGB weights to 64 channels.
       - Cite solution_lesson_node_00030: Native adaptation prevents gradient explosion compared to custom stems.
    3. Output: Single logit for binary classification.
    """

    def __init__(self):
        super(MGMTNet, self).__init__()

        # Retrieve hyperparameters from config
        in_channels = config.IN_CHANNELS
        backbone_name = config.BACKBONE_NAME

        # Backbone
        # EfficientNet-B0 initialized with pretrained weights.
        # Modified to accept 'in_channels' (64) directly.
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=in_channels, num_classes=1
        )

    def forward(self, x):
        # x shape: (B, 64, 224, 224)

        # Apply Backbone and Classification Head
        # Output shape: (B, 1)
        x = self.backbone(x)

        return x
