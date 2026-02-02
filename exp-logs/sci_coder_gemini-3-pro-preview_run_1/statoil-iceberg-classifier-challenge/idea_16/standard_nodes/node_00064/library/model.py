import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from library.config import Config


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture for Iceberg vs Ship classification.
    Implements Late Fusion of incidence angle and Global Average Pooling.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): If True, loads ImageNet weights.
        """
        super(IcebergResNet, self).__init__()

        # Load ResNet-18 backbone
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        original_model = resnet18(weights=weights)

        # Input channels check: ResNet expects 3 channels by default.
        # Our data is 3 channels (Band1, Band2, Avg), so no modification needed for conv1.

        # Feature Extractor: Remove the fully connected layer and the pooling layer
        # We keep everything up to the final pooling layer
        # list(original_model.children())[:-2] gives layers up to layer4
        self.features = nn.Sequential(*list(original_model.children())[:-2])

        # Global Average Pooling
        # Explicitly defined to reject Max Pooling as per Lesson 00019
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension for ResNet18 is 512
        self.num_features = 512

        # Late Fusion Dimension: 512 (Image) + 1 (Angle)
        self.fusion_dim = self.num_features + 1

        # Minimalist Head
        # BN -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.fusion_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass with Late Fusion.

        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W).
            angle (torch.Tensor): Incidence angle tensor of shape (B,).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # 1. Feature Extraction
        # Shape: (B, 512, H/32, W/32) -> (B, 512, 7, 7) for 224x224 input
        x = self.features(x)

        # 2. Global Average Pooling
        # Shape: (B, 512, 1, 1)
        x = self.avgpool(x)

        # 3. Flatten
        # Shape: (B, 512)
        x = torch.flatten(x, 1)

        # 4. Late Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)
        # Concatenate: (B, 512) + (B, 1) -> (B, 513)
        x = torch.cat([x, angle], dim=1)

        # 5. Classification Head
        # Shape: (B, 1)
        x = self.head(x)

        return x
