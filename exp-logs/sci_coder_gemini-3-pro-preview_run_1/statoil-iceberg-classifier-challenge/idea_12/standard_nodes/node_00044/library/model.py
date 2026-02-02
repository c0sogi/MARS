import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model with Late Fusion for Iceberg/Ship classification.

    Architecture:
    1. Backbone: ResNet-18 (Pretrained on ImageNet)
    2. Pooling: Global Average Pooling (Native to ResNet)
    3. Fusion: Concatenation of 512-dim image features + 1-dim angle
    4. Head: BatchNorm -> Dropout -> Linear(513, 1)
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # 1. Load Backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        base_model = models.resnet18(weights=weights)

        # Remove the final fully connected layer (fc)
        # We keep the AdaptiveAvgPool2d which is the second to last layer
        # list(children())[:-1] returns all layers except the last FC layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # 2. Define Fusion and Head
        # Image features (512) + Angle (1) = 513
        input_features = 512 + 1

        self.head = nn.Sequential(
            nn.BatchNorm1d(input_features),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(input_features, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).
            angle (torch.Tensor): Input incidence angles of shape (B,) or (B, 1).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # Feature Extraction
        # Output shape: (B, 512, 1, 1)
        features = self.backbone(x)

        # Flatten: (B, 512)
        features = features.view(features.size(0), -1)

        # Ensure angle is (B, 1)
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # Late Fusion
        # Concatenate image features and angle
        combined = torch.cat((features, angle), dim=1)

        # Classification
        logits = self.head(combined)

        return logits
