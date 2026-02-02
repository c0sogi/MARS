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
        # Pass in_chans to match the 9-channel input from the data pipeline
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=DROPOUT_RATE,
            in_chans=INPUT_CHANNELS,
        )

    def forward(self, x):
        # Pass through the backbone
        x = self.backbone(x)
        return x
