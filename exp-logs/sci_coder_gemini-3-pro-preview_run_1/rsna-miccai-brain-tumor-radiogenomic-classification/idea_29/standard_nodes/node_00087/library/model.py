import torch
import torch.nn as nn
import timm
from library.config import Config


class SDWIVNet(nn.Module):
    """
    Standard 2.5D EfficientNet-B0.

    Reverted from SDWIVNet to a simpler baseline to leverage ImageNet priors effectively.
    Accepts 3-channel input (FLAIR, T1wCE, T2w center slices).
    """

    def __init__(self):
        super().__init__()

        # Initialize EfficientNet-B0 backbone
        # drop_rate controls the dropout before the final classifier layer
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=Config.NUM_CLASSES,
            drop_rate=Config.CLASSIFIER_DROPOUT,
        )
        # No modification to first layer needed as Config.NUM_CHANNELS is 3

    def forward(self, x):
        """
        Forward pass.
        x: Input tensor of shape (Batch, 3, Height, Width)
        """
        return self.backbone(x)
