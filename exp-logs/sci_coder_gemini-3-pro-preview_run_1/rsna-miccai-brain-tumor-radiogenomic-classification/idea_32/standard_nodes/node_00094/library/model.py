import torch
import torch.nn as nn
import timm
from library.config import BACKBONE, NUM_CLASSES, DROPOUT_RATE, INPUT_CHANNELS, SEED


class RNVSNetwork(nn.Module):
    """
    Simplified Network for 3-channel input (1 slice x 3 modalities).
    Uses standard EfficientNet-B0 with ImageNet weights.
    """

    def __init__(
        self,
        backbone_name=BACKBONE,
        pretrained=True,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
        input_dropout_prob=0.0,  # Kept for API compatibility but unused
    ):
        super(RNVSNetwork, self).__init__()

        # 1. Create Backbone
        # efficientnet_b0 expects 3 channels by default, which matches our new config
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout_rate,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        x = self.backbone(x)
        return x
