import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Ship vs. Iceberg classification.

    Implements the 'Full-Fit ResNet-18 Seed Ensemble with Corrected Composite Fusion' architecture:
    1. Backbone: ResNet-18 pretrained on ImageNet.
    2. Pooling: Global Average Pooling (GAP).
    3. Late Fusion: Concatenates the 512-dim image feature vector with the normalized scalar incidence angle.
    4. Head: Minimalist design (BatchNorm -> Dropout -> Linear).
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load Pretrained ResNet-18
        # We use the explicit weights parameter as per modern torchvision standards
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)

        # The output feature dimension of ResNet-18 after Global Average Pooling is 512
        self.backbone_dim = 512

        # Fusion Dimension: Backbone Features (512) + Incidence Angle (1)
        self.fusion_dim = self.backbone_dim + 1

        # Minimalist Classification Head
        # Constraints: BatchNorm, Dropout(0.5), Single Linear Layer
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.fusion_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W).
            angle (torch.Tensor): Normalized incidence angles of shape (Batch,) or (Batch, 1).

        Returns:
            torch.Tensor: Raw logits of shape (Batch, 1).
        """
        # --- Backbone Forward Pass ---
        # We execute the ResNet layers manually to intercept features before the original FC layer

        # Stem
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # Residual Blocks
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # Global Average Pooling
        # Output shape: (Batch, 512, 1, 1)
        x = self.resnet.avgpool(x)

        # Flatten to vector
        # Output shape: (Batch, 512)
        x = torch.flatten(x, 1)

        # --- Late Fusion ---
        # Ensure angle tensor has the correct shape (Batch, 1) for concatenation
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate image features with the scalar angle
        # Output shape: (Batch, 513)
        x = torch.cat((x, angle), dim=1)

        # --- Classification Head ---
        # Pass fused vector through the minimalist head
        x = self.head(x)

        return x
