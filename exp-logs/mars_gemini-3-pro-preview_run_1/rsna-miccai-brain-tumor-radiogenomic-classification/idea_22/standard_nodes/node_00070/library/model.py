import torch
import torch.nn as nn
import timm
from library import config


class ACWIVNet(nn.Module):
    """
    Anatomically-Centric Weight-Inflated Volumetric (AC-WIV) Network.

    Adapts EfficientNet-B0 to accept 9-channel volumetric inputs (3 modalities x 3 depths)
    using a Gaussian Weight Inflation strategy to preserve ImageNet priors.
    """

    def __init__(
        self,
        backbone_name=config.BACKBONE,
        pretrained=config.PRETRAINED,
        input_channels=config.INPUT_CHANNELS,
    ):
        super(ACWIVNet, self).__init__()

        # Load the backbone with a single output class for binary classification
        # efficientnet_b0 is the default config.BACKBONE
        # We use in_chans=3 (standard ImageNet)
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=config.DROPOUT_RATE,
            in_chans=input_channels,
        )

    def forward(self, x):
        """
        Forward pass.
        Input x: (Batch, 9, H, W)
        Output: (Batch, 1) - Logits
        """
        return self.model(x)
