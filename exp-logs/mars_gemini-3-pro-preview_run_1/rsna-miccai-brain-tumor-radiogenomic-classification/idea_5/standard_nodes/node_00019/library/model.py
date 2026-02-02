import torch
import torch.nn as nn
import timm
from library.config import Config


class GlioblastomaModel(nn.Module):
    """
    Standard 2.5D CNN (EfficientNet-B0) processing a single multi-channel slice.

    Cite Lesson 18: Reverted from Siamese network to avoid redundancy penalty.
    Cite Lesson 2: Middle slice selection (handled in dataset).
    """

    def __init__(self, pretrained=True):
        super(GlioblastomaModel, self).__init__()

        # Load EfficientNet-B0 from timm
        # num_classes=1 creates the classifier head directly
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=1,
            in_chans=3,
            drop_rate=Config.DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        return self.backbone(x)
