import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Iceberg vs. Ship classification.

    Architecture:
    1. Backbone: ResNet-18 pretrained on ImageNet.
    2. Pooling: Global Average Pooling (GAP) to obtain a 512-dim vector.
    3. Fusion: Late Fusion of the normalized incidence angle (scalar).
    4. Head: Minimalist head (BatchNorm -> Dropout -> Linear).
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet-18 weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)

        # Remove the original fully connected layer and average pooling layer
        # as we will define our own pooling and head logic.
        del self.backbone.fc
        del self.backbone.avgpool

        # Define Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimensions
        # ResNet-18 layer4 outputs 512 channels
        self.num_image_features = 512
        self.num_angle_features = 1

        # Combined dimension for the dense layer
        self.fusion_dim = self.num_image_features + self.num_angle_features

        # Minimalist Classification Head
        # As per strategy: BatchNorm -> Dropout(0.5) -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.fusion_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (Batch_Size, 3, H, W).
            angle (torch.Tensor): Input incidence angles of shape (Batch_Size,) or (Batch_Size, 1).

        Returns:
            torch.Tensor: Raw logits of shape (Batch_Size, 1).
        """
        # --- Backbone Feature Extraction ---
        # We manually call the layers to skip the deleted fc/avgpool
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # --- Global Average Pooling ---
        x = self.global_pool(x)
        x = torch.flatten(x, 1)  # Shape: (B, 512)

        # --- Late Fusion ---
        # Ensure angle tensor has shape (B, 1) for concatenation
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Concatenate image features with the scalar angle
        x = torch.cat((x, angle), dim=1)  # Shape: (B, 513)

        # --- Classification Head ---
        logits = self.head(x)

        return logits
