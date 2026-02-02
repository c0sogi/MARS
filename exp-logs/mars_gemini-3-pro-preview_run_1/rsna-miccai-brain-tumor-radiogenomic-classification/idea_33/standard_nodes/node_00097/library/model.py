import torch
import torch.nn as nn
import timm
from library.config import (
    INPUT_CHANNELS,
    NUM_CLASSES,
    DROPOUT_RATE,
    INPUT_DROPOUT_PROB,
    BACKBONE,
)


class RARVEfficientNet(nn.Module):
    def __init__(self):
        super(RARVEfficientNet, self).__init__()

        # Initialize the backbone using timm
        # efficientnet_b0 is used as per configuration
        # Standard input is 3 channels, which matches our configuration now.
        self.backbone = timm.create_model(
            BACKBONE, pretrained=True, num_classes=NUM_CLASSES, drop_rate=DROPOUT_RATE
        )

    def forward(self, x):
        # Pass through the backbone
        x = self.backbone(x)
        return x
