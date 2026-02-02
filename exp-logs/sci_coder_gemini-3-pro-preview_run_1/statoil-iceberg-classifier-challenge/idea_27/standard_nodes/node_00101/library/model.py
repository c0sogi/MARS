import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from library.config import FUSION_DIM, DROPOUT_RATE, NUM_CLASSES


class IcebergResNet(nn.Module):
    """
    ResNet-18 based model with Late Fusion for Iceberg detection.

    Architecture:
    1. Backbone: ResNet-18 (Pretrained on ImageNet)
    2. Pooling: Global Average Pooling (GAP) -> 512 dim
    3. Fusion: Concat(GAP, inc_angle) -> 513 dim
    4. Head: BatchNorm -> Dropout -> Linear -> Logit
    """

    def __init__(self, dropout_rate=DROPOUT_RATE):
        """
        Args:
            dropout_rate (float): Probability of an element to be zeroed in the dropout layer.
        """
        super(IcebergResNet, self).__init__()

        # Initialize Backbone
        # We use the default ImageNet weights (IMAGENET1K_V1)
        # This provides a robust feature extractor for the 3-channel input
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Feature Extractor Construction
        # ResNet18 layers: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We retain everything up to 'avgpool' to get the 512-dim feature vector.
        # We remove the final 'fc' layer.
        modules = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        # Minimalist Classification Head (Late Fusion)
        # Input dimension = 512 (Image Features) + 1 (Incidence Angle)
        input_dim = FUSION_DIM + 1

        self.head = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(input_dim, NUM_CLASSES),
        )

    def forward(self, x, inc_angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image batch of shape (B, 3, H, W).
            inc_angle (torch.Tensor): Incidence angle batch of shape (B,) or (B, 1).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # 1. Feature Extraction
        # Pass images through ResNet backbone
        # Output shape: (B, 512, 1, 1)
        features = self.feature_extractor(x)

        # 2. Flatten
        # Flatten spatial dimensions to get the feature vector
        # Output shape: (B, 512)
        features = torch.flatten(features, 1)

        # 3. Late Fusion
        # Ensure inc_angle has the correct shape (B, 1) for concatenation
        if inc_angle.dim() == 1:
            inc_angle = inc_angle.unsqueeze(1)

        # Concatenate image features with the scalar incidence angle
        # Output shape: (B, 513)
        fused = torch.cat((features, inc_angle), dim=1)

        # 4. Classification Head
        # Pass fused vector through BN -> Dropout -> Linear
        # Output shape: (B, 1)
        logits = self.head(fused)

        return logits
