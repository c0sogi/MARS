import torch
import torch.nn as nn
import timm
from library import config


class SIRVEfficientNet(nn.Module):
    """
    Standard EfficientNet-B0 backbone.
    Cite solution_lesson_node_00009: Avoid naively stacking volumetric depth slices.
    We revert to a standard 3-channel input (FLAIR, T1wCE, T2w) to leverage
    pretrained ImageNet priors effectively.
    """

    def __init__(self):
        super().__init__()

        # Create backbone
        # drop_rate sets the dropout rate before the final classifier layer
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=config.PRETRAINED,
            num_classes=config.NUM_CLASSES,
            drop_rate=config.DROPOUT_RATE,
            in_chans=config.NUM_CHANNELS,  # Should be 3
        )

    def forward(self, x):
        # Forward pass through backbone
        x = self.backbone(x)
        return x
