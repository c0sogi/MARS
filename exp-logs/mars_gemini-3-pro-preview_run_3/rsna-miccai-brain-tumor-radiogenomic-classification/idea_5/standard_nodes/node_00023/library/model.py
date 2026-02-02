import torch
import torch.nn as nn
import timm


from library.config import Config


class BraTS25DNet(nn.Module):
    """
    2.5D Stacked CNN (Cite solution_lesson_node_00022).
    Stacks volumetric slices into the channel dimension to allow a 2D CNN
    to process 3D data efficiently while preserving depth context in the
    feature extraction phase.
    """

    def __init__(self, num_classes=1, pretrained=True):
        super(BraTS25DNet, self).__init__()

        # Calculate input channels: 4 modalities * 16 slices = 64
        # Cite solution_lesson_node_00018: Early Fusion via channel stacking
        in_chans = Config.NUM_CHANNELS * Config.NUM_SLICES

        # Use EfficientNet-B0
        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 4, D, H, W)
        Returns:
            torch.Tensor: Logits of shape (B, num_classes)
        """
        # Flatten Modalities and Depth into Channels
        # (B, 4, 16, 256, 256) -> (B, 64, 256, 256)
        B, C, D, H, W = x.shape
        x = x.view(B, C * D, H, W)

        return self.encoder(x)
