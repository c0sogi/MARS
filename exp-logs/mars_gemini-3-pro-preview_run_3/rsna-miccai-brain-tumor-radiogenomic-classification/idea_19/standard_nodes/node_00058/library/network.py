import torch
import torch.nn as nn
import timm
from library.config import Config


class SHDNet(nn.Module):
    """
    Standardized 2.5D Network.

    Adapts EfficientNet-B0 to accept 64 input channels (16 slices * 4 modalities) directly.
    """

    def __init__(self, drop_path_rate=None):
        super(SHDNet, self).__init__()

        # Use config default if not provided
        if drop_path_rate is None:
            drop_path_rate = Config.DROP_PATH_RATE

        # Cite solution_lesson_node_00038: Use library's native head for proper regularization (Dropout)
        # Cite solution_lesson_node_00034: 64 channels is stable for direct adaptation, no stem needed
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=64,
            drop_path_rate=drop_path_rate,
            num_classes=1,
            drop_rate=0.2,
        )

    def forward(self, x):
        # x: (B, 64, 224, 224) -> (B, 1)
        return self.backbone(x)
