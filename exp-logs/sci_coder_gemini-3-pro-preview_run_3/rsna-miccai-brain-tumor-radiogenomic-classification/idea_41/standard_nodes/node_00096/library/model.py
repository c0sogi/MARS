import torch
import torch.nn as nn
import timm
from library.config import BACKBONE, IN_CHANS, NUM_CLASSES, DROP_PATH_RATE


class BraTSModel(nn.Module):
    """
    2.5D Stacked EfficientNet.

    This architecture processes a 3D MRI volume by stacking slices along the channel dimension.
    Input: (B, 64, 224, 224) -> 16 slices * 4 modalities.
    """

    def __init__(self):
        super().__init__()

        # Initialize the backbone using timm
        # We use num_classes=NUM_CLASSES directly to preserve the library's
        # default classifier structure and regularization (dropout).
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=True,
            in_chans=IN_CHANS,
            num_classes=NUM_CLASSES,
            drop_path_rate=DROP_PATH_RATE,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Tensor of shape (B, 64, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        return self.backbone(x)
