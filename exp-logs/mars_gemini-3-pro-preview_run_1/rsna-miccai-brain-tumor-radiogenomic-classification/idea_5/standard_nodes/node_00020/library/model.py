import torch
import torch.nn as nn
import timm
from library.config import Config


class SiameseEfficientNet(nn.Module):
    """
    Siamese 2.5D CNN (EfficientNet-B0) processing multiple views (slices).
    """

    def __init__(self, pretrained=True):
        super(SiameseEfficientNet, self).__init__()

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
            x (torch.Tensor): Input tensor of shape (Batch, Views, Channels, Height, Width).
        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # Handle 5D input: (Batch, Views, Channels, Height, Width)
        if x.dim() == 5:
            b, v, c, h, w = x.shape
            # Merge batch and views for processing
            x = x.view(b * v, c, h, w)
            logits = self.backbone(x)  # (Batch * Views, 1)
            # Reshape back to separate views
            logits = logits.view(b, v)
            # Average pooling across views (Siamese aggregation)
            logits = torch.mean(logits, dim=1, keepdim=True)  # (Batch, 1)
            return logits

        return self.backbone(x)
