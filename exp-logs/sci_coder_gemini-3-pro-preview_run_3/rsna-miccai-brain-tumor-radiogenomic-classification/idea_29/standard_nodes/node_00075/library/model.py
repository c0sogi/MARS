import torch
import torch.nn as nn
import timm
from library.config import Config


class VAMSHDNet(nn.Module):
    """
    VAMS-HD Network (Optimized).

    Architecture:
    1. Input: (B, 64, 224, 224) - 16 slices * 4 modalities.
    2. Backbone: EfficientNet-B0 (timm)
       - in_chans=64 (Native Adaptation, Cite solution_lesson_node_00030)
       - drop_path_rate=0.2
    3. Head: Global Average Pooling + Linear (Logits)
    """

    def __init__(self):
        super(VAMSHDNet, self).__init__()

        # Native Input Adaptation (Cite solution_lesson_node_00030)
        # We remove the custom stem and rely on timm's robust weight recycling.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            in_chans=Config.IN_CHANNELS,
            num_classes=Config.NUM_CLASSES,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass of the VAMS-HD Network.
        Args:
            x (torch.Tensor): Input tensor of shape (B, 64, 224, 224)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # Pass directly to backbone
        x = self.backbone(x)

        return x
