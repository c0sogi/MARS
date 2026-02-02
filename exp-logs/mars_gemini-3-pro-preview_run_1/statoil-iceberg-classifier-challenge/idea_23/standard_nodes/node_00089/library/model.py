import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IcebergResNet(nn.Module):
    """
    ResNet-18 based model for Iceberg vs Ship classification.

    Architecture:
    1. Backbone: Pretrained ResNet-18 (ImageNet weights).
    2. Pooling: Global Average Pooling (GAP).
    3. Fusion: Late fusion of scalar incidence angle with GAP features.
    4. Head: Minimalist classifier (BatchNorm -> Dropout -> Linear).
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load pretrained ResNet-18
        # We use 'DEFAULT' weights which corresponds to the best available ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT if Config.PRETRAINED else None
        resnet = models.resnet18(weights=weights)

        # Extract feature extractor (remove avgpool and fc layers)
        # ResNet-18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # We keep everything up to layer4
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Global Average Pooling
        # Reduces (Batch, 512, 7, 7) -> (Batch, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion and Classification Head
        # Input dimension = 512 (ResNet features) + 1 (Incidence Angle)
        self.fusion_dim = Config.FUSION_DIM  # 513

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
            angle (torch.Tensor): Incidence angles of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Feature Extraction
        # Shape: (Batch, 512, H/32, W/32) -> (Batch, 512, 7, 7) for 224x224 input
        x = self.features(x)

        # 2. Global Average Pooling
        # Shape: (Batch, 512, 1, 1)
        x = self.avgpool(x)

        # Flatten
        # Shape: (Batch, 512)
        x = torch.flatten(x, 1)

        # 3. Late Fusion
        # Ensure angle is (Batch, 1)
        angle = angle.view(-1, 1)

        # Concatenate features and angle
        # Shape: (Batch, 513)
        x = torch.cat((x, angle), dim=1)

        # 4. Classification Head
        # Shape: (Batch, 1)
        logits = self.head(x)

        return logits
