import torch
import torch.nn as nn
import timm
from library.config import Config


class Stabilized25DNet(nn.Module):
    """
    Stabilized 2.5D Network using timm.

    1. Input: (B, 64, 224, 224) - 16 slices * 4 modalities.
    2. Backbone: EfficientNet-B0 (pretrained), with first layer adapted by timm.
    3. Head: Linear layer for binary classification (logits).
    """

    def __init__(self):
        super(Stabilized25DNet, self).__init__()

        # Cite solution_lesson_node_00030: Use library native adaptation for input channels
        # Cite solution_lesson_node_00034: Avoid excessive channel depth (staying at 64)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=Config.IN_CHANNELS,
            num_classes=1,
            drop_rate=0.2,
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 64, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
