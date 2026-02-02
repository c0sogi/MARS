import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class MultiViewResNet(nn.Module):
    """
    Multi-View ResNet-18 model for bird species classification.
    Handles 5D input (Batch, Tiles, Channels, Height, Width) by processing tiles independently and aggregating.
    """

    def __init__(self):
        super(MultiViewResNet, self).__init__()

        # Load Pretrained ResNet18
        weights = "DEFAULT" if Config.PRETRAINED else None
        self.backbone = models.resnet18(weights=weights)

        # Modify the final Fully Connected layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, Config.NUM_CLASSES)

    def forward(self, x):
        # Handle 5D input: (Batch, Tiles, Channels, Height, Width)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            # Flatten Batch and Tiles dimensions
            x = x.view(B * T, C, H, W)

            # Forward pass through backbone
            out = self.backbone(x)  # Shape: (B * T, NumClasses)

            # Reshape back to separate Batch and Tiles
            out = out.view(B, T, -1)

            # Aggregate predictions across tiles (Mean Pooling)
            out = torch.mean(out, dim=1)
        else:
            # Fallback for 4D input
            out = self.backbone(x)

        return out
