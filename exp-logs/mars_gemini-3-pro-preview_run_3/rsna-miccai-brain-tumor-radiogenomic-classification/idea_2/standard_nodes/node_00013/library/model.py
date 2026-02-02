import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class Stacked25DNet(nn.Module):
    """
    2.5D Stacked CNN Network.
    Cite solution_lesson_node_00010: Prefer 2.5D Channel Stacking over MIL for small datasets.

    Architecture:
    1. Feature Extractor: EfficientNet-B0 (pretrained).
       Input channels = Num_Slices * Num_Modalities.
    2. Classifier: Linear layer to predict MGMT_value.
    """

    def __init__(self, in_channels=128):
        super(Stacked25DNet, self).__init__()

        # Use timm to create EfficientNet-B0.
        # in_chans adapts the first conv layer to accept the stacked volume.
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=in_channels, num_classes=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, H, W).
                              Channels = Num_Slices * Modalities.

        Returns:
            logits (torch.Tensor): Output logits of shape (Batch_Size, 1).
        """
        return self.model(x)
