import torch
import torch.nn as nn
import timm
from library.config import Config


class S3HDNetwork(nn.Module):
    """
    Semantically-Structured Stabilized High-Density (S3HD) Network.

    Architecture:
    1. Input: (B, 128, 224, 224) - 32 slices x 4 modalities.
    2. Stabilized Global-Mixing Stem: Compresses 128ch -> 64ch with He Init.
    3. Backbone: EfficientNet-B0 (in_chans=64, drop_path=0.2).
    4. Head: Global Average Pooling + FC (Logits).
    """

    def __init__(self):
        super(S3HDNetwork, self).__init__()

        # Input channels from Config (expected 64)
        input_channels = Config.INPUT_CHANNELS

        # Backbone: EfficientNet-B0
        # We use timm's native in_chans adaptation which is more stable than custom stems
        # for moderate channel counts (Cite solution_lesson_node_00030)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,  # efficientnet_b0
            pretrained=True,
            in_chans=input_channels,  # 64
            num_classes=Config.NUM_CLASSES,  # 1
            drop_path_rate=Config.DROP_PATH_RATE,  # 0.2
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 64, 224, 224).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Pass directly through Backbone
        # Shape: (B, 64, 224, 224) -> (B, 1)
        logits = self.backbone(x)

        return logits
