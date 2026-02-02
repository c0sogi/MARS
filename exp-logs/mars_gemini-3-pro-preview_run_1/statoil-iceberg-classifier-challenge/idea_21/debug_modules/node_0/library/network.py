import torch
import torch.nn as nn
from torchvision import models
from library.config import DROPOUT_RATE


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture for Iceberg detection with Late Fusion of incidence angle.

    Architecture:
    1. Backbone: ResNet-18 (Pretrained on ImageNet)
    2. Pooling: Global Average Pooling (GAP)
    3. Fusion: Concatenation of 512-dim image features with 1-dim normalized incidence angle.
    4. Head: BatchNorm -> Dropout -> Linear
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load pretrained ResNet18
        # We use the default weights (ImageNet)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.resnet = models.resnet18(weights=weights)

        # Extract feature extractor (conv1 -> layer4)
        # We exclude the original avgpool and fc layer
        self.features = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        )

        # Global Average Pooling
        # Reduces (B, 512, H, W) -> (B, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension for ResNet18
        self.num_features = 512

        # Fusion dimension: Image features + 1 angle scalar
        self.fusion_dim = self.num_features + 1

        # Minimalist Classification Head
        # As per design: Batch Normalization -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(self.fusion_dim, 1),
        )

    def forward(self, x, angle):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images, shape (Batch, 3, Height, Width)
            angle (torch.Tensor): Normalized incidence angles, shape (Batch,) or (Batch, 1)

        Returns:
            torch.Tensor: Logits, shape (Batch, 1)
        """
        # 1. Feature Extraction
        x = self.features(x)

        # 2. Global Average Pooling
        x = self.avgpool(x)

        # 3. Flatten
        x = torch.flatten(x, 1)  # (Batch, 512)

        # 4. Process Angle for Concatenation
        # Ensure angle is (Batch, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # 5. Late Fusion
        x = torch.cat((x, angle), dim=1)  # (Batch, 513)

        # 6. Classification Head
        logits = self.head(x)

        return logits
