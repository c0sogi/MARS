import torch
import torch.nn as nn
import timm
from library.config import Config


class WIVSNet(nn.Module):
    """
    Standard EfficientNet-B0 backbone processing a 3-channel input (FLAIR, T1wCE, T2w).
    Reverts to the robust 2.5D baseline.
    """

    def __init__(self, pretrained=True):
        super(WIVSNet, self).__init__()

        # Initialize standard EfficientNet-B0 with 3 channels
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=1,
            drop_rate=Config.DROPOUT,
            in_chans=3,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        return self.backbone(x)
