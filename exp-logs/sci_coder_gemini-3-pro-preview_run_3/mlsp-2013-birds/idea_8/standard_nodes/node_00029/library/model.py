import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class MultiViewResNet(nn.Module):
    """
    A Multi-Instance Learning (MIL) model based on ResNet-18.

    It processes multiple temporal tiles (views) of a spectrogram independently
    through a shared backbone and aggregates the results using Max-Pooling.
    """

    def __init__(self):
        super(MultiViewResNet, self).__init__()

        # Load Pretrained ResNet18
        # Using "DEFAULT" loads the best available weights (ImageNet)
        weights = "DEFAULT" if Config.PRETRAINED else None
        self.backbone = models.resnet18(weights=weights)

        # Modify the final Fully Connected layer
        # The default ResNet18 fc layer outputs 1000 classes (ImageNet).
        # We replace it to output Config.NUM_CLASSES (19 species).
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass for Multi-View processing.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Num_Tiles, Channels, Height, Width).
                              e.g., (B, 3, 3, 224, 224).

        Returns:
            torch.Tensor: Aggregated logits of shape (Batch, Num_Classes).
        """
        # 1. Unpack dimensions
        bs, tiles, c, h, w = x.size()

        # 2. Collapse Batch and Tile dimensions
        # We treat every tile as an independent image for the CNN backbone.
        # New shape: (Batch * Num_Tiles, C, H, W)
        x = x.view(bs * tiles, c, h, w)

        # 3. Pass through Backbone
        # The backbone outputs logits for each tile.
        # Output shape: (Batch * Num_Tiles, Num_Classes)
        tile_logits = self.backbone(x)

        # 4. Reshape back to separate Batch and Tiles
        # Shape: (Batch, Num_Tiles, Num_Classes)
        tile_logits = tile_logits.view(bs, tiles, -1)

        # 5. Multi-Instance Aggregation (Max Pooling)
        # We take the maximum logit across the temporal tiles for each class.
        # Logic: If a bird is present in ANY time segment (tile), the bag (recording) is positive.
        # Shape: (Batch, Num_Classes)
        bag_logits, _ = torch.max(tile_logits, dim=1)

        return bag_logits
