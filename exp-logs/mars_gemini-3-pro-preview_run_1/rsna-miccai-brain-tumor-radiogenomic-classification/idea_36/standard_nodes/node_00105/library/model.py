import torch
import torch.nn as nn
import timm

# Import configuration from the provided library
from library.config import (
    BACKBONE,
    NUM_CLASSES,
    DROPOUT_RATE,
    IN_CHANNELS,
)


class CASIVNet(nn.Module):
    """
    Standard EfficientNet-B0 for 3-channel input (FLAIR, T1wCE, T2w).
    Reverts to standard transfer learning as per Lesson 00009 (Avoid naive channel stacking)
    and Lesson 00025 (Avoid learnable projections/complex adaptations).
    """

    def __init__(self):
        super(CASIVNet, self).__init__()

        # 1. Initialize Backbone
        # We use timm to load the EfficientNet-B0 with Noisy Student weights (ns)
        # drop_rate sets the dropout probability before the final classifier
        # IN_CHANNELS is 3, which matches the pretrained weights (RGB)
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=DROPOUT_RATE,
            in_chans=IN_CHANNELS,
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 9, H, W)
        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        return self.backbone(x)
