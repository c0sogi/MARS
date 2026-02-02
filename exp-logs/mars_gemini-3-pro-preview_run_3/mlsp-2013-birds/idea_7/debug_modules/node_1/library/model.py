import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class MILResNet18(nn.Module):
    """
    Multi-Instance Learning model using a ResNet-18 backbone.
    Processes multiple temporal tiles (instances) per recording (bag) and aggregates
    predictions using Max-Pooling.
    """

    def __init__(self):
        super(MILResNet18, self).__init__()

        # Load Pretrained ResNet18
        # Using the updated weights API for torchvision
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None

        self.backbone = models.resnet18(weights=weights)

        # Modify the final fully connected layer to match the number of classes
        # ResNet18's fc layer has 512 input features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass for the MIL model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Tiles, Channels, Height, Width).
                              Tiles correspond to temporal segments of the spectrogram.

        Returns:
            torch.Tensor: Aggregated logits of shape (Batch, Num_Classes).
        """
        # Unpack dimensions
        # B: Batch size
        # T: Number of tiles (instances)
        # C: Channels (3, replicated)
        # H: Height (224)
        # W: Width (224)
        B, T, C, H, W = x.shape

        # Collapse Batch and Tile dimensions to process instances independently via the backbone
        # New shape: (B * T, C, H, W)
        x = x.view(B * T, C, H, W)

        # Pass through the ResNet backbone
        # Output shape: (B * T, Num_Classes)
        x = self.backbone(x)

        # Reshape back to separate Batch and Tile dimensions
        # Shape: (B, T, Num_Classes)
        x = x.view(B, T, Config.NUM_CLASSES)

        # MIL Aggregation: Max-Pooling across the Tile dimension
        # We take the maximum logit value across all tiles for each class.
        # This assumes that if a bird is present in ANY tile, the bag is positive.
        # Shape: (B, Num_Classes)
        x, _ = torch.max(x, dim=1)

        return x
