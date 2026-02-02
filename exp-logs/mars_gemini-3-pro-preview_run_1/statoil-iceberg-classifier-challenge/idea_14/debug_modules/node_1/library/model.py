import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based architecture for Ship vs. Iceberg classification.

    Features:
    - Backbone: ResNet-18 (pretrained on ImageNet).
    - Pooling: Global Average Pooling (GAP).
    - Fusion: Late fusion of image features with scalar incidence angle.
    - Head: Minimalist head (BN -> Dropout -> Linear).
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # 1. Load Backbone
        # We use the 'DEFAULT' weights which correspond to the best available ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT if Config.USE_PRETRAINED else None
        resnet = models.resnet18(weights=weights)

        # 2. Extract Feature Extractor
        # We keep everything up to the final FC layer.
        # ResNet structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
        # We will use the layers explicitly to have control over the pooling.
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        # 3. Global Average Pooling
        # Reduces (B, 512, 7, 7) -> (B, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 4. Classification Head
        # Input dimension: 512 (from image) + 1 (from angle) = 513
        self.input_dim = 512 + 1

        # Minimalist Head: BN -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.input_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W).
            angle (torch.Tensor): Incidence angle tensor of shape (B,).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # 1. Feature Extraction
        x = self.features(x)  # Shape: (B, 512, 7, 7)

        # 2. Pooling
        x = self.avgpool(x)  # Shape: (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (B, 512)

        # 3. Late Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate image features and angle
        x = torch.cat((x, angle), dim=1)  # Shape: (B, 513)

        # 4. Classification Head
        logits = self.head(x)  # Shape: (B, 1)

        return logits
