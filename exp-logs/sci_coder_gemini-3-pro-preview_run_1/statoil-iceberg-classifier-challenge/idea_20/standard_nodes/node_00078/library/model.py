import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    Standard ResNet-18 based architecture with Late Fusion for incidence angle.
    Geometric invariance is handled via Test-Time Augmentation (TTA), not in the forward pass.
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet18
        # Using V1 weights as standard for ImageNet pretraining
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        original_model = models.resnet18(weights=weights)

        # Extract backbone: keep everything up to the Average Pooling layer
        self.backbone = nn.Sequential(*list(original_model.children())[:-1])

        # Feature dimension from ResNet18 GAP is 512
        self.feature_dim = 512

        # Angle normalization statistics (derived from dataset analysis)
        # Mean: 39.2829, Std: 3.8362
        self.angle_mean = 39.2829
        self.angle_std = 3.8362

        # Minimalist Head for Late Fusion
        # Input: 512 (Image Features) + 1 (Normalized Angle)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim + 1),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.feature_dim + 1, 1),
        )

    def forward(self, x, angle):
        """
        Standard forward pass.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, 224, 224)
            angle (torch.Tensor): Incidence angles of shape (B,) or (B, 1)

        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Backbone Feature Extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # (B, 512)

        # 2. Angle Processing & Late Fusion
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Normalize angle using global stats
        angle_norm = (angle - self.angle_mean) / self.angle_std

        # Concatenate image features and angle
        fused_features = torch.cat([features, angle_norm], dim=1)  # (B, 513)

        # 3. Classification Head
        logits = self.head(fused_features)  # (B, 1)

        return logits
