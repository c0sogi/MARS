import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class BraTS25DNet(nn.Module):
    """
    2.5D Stacked CNN (Cite solution_lesson_node_00010).
    Takes a volume stacked in the channel dimension and processes it with a 2D CNN.
    """

    def __init__(self, in_channels=64):
        super(BraTS25DNet, self).__init__()
        # EfficientNet-B0 with modified input channels
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=in_channels, num_classes=1
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, In_Channels, H, W).
        Returns:
            logits (torch.Tensor): Output logits of shape (Batch_Size, 1).
        """
        return self.backbone(x)
