import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class MILResNet18(nn.Module):
    """
    Multi-Instance Learning model using a ResNet-18 backbone.

    Architecture:
    1. Input: (Batch, Num_Tiles, Channels, Height, Width)
    2. Backbone: ResNet-18 (shared across tiles)
    3. Aggregation: Max-Pooling of logits across tiles
    4. Output: Bag-level logits (Batch, Num_Classes)
    """

    def __init__(self):
        super(MILResNet18, self).__init__()

        # Load ResNet-18 backbone
        # Using the new weights API if available, or falling back to pretrained=True logic if needed
        # Config.PRETRAINED is a boolean
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # Replace the final fully connected layer
        # ResNet18 fc layer input features is typically 512
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Num_Tiles, Channels, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        # Unpack dimensions
        batch_size, num_tiles, c, h, w = x.size()

        # Collapse Batch and Num_Tiles dimensions to process all images in parallel
        # New shape: (Batch * Num_Tiles, Channels, Height, Width)
        x_reshaped = x.view(batch_size * num_tiles, c, h, w)

        # Pass through the backbone
        # Output shape: (Batch * Num_Tiles, Num_Classes)
        tile_logits = self.backbone(x_reshaped)

        # Reshape back to separate Batch and Num_Tiles
        # Shape: (Batch, Num_Tiles, Num_Classes)
        tile_logits = tile_logits.view(batch_size, num_tiles, Config.NUM_CLASSES)

        # Multi-Instance Aggregation: Max Pooling
        # We take the max logit across the tiles for each class.
        # This implies: "If this bird is present in ANY tile, the bag score is high."
        # Shape: (Batch, Num_Classes)
        bag_logits, _ = torch.max(tile_logits, dim=1)

        return bag_logits
