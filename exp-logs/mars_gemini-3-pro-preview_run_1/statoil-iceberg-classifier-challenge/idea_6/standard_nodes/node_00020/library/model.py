import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class IcebergResNet(nn.Module):
    """
    Stratified Dual-Stream ResNet-18 with Late Fusion.

    Architecture:
    1. Backbone: ResNet-18 (Pretrained on ImageNet).
    2. Pooling: Dual-Stream (Global Average + Global Max).
    3. Fusion: Concatenation of image features with incidence angle.
    4. Head: BatchNorm -> Dropout -> Dense -> Logits.
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load pretrained ResNet-18
        # Using the new weights API if available, else fallback to pretrained=True logic implicitly
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            resnet = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            resnet = models.resnet18(pretrained=True)

        # Remove the fully connected layer and the average pooling layer
        # We keep the convolutional layers (conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
        # list(resnet.children())[:-2] gets layers up to layer4
        self.features = nn.Sequential(*list(resnet.children())[:-2])

        # Feature dimension for ResNet-18 at layer4 is 512
        self.feature_dim = 512

        # Dual-Stream Pooling results in 512 * 2 = 1024 dimensions
        # Adding 1 dimension for the incidence angle
        self.fusion_dim = (self.feature_dim * 2) + 1

        # Classification Head
        # As per design: BatchNorm -> Dropout -> Dense
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.fusion_dim, 1),  # Binary classification (logits)
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W).
            angle (torch.Tensor): Incidence angle tensor of shape (B,).

        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # 1. Feature Extraction
        # Output shape: (B, 512, 7, 7) for 224x224 input
        x = self.features(x)

        # 2. Dual-Stream Pooling
        # Global Average Pooling -> (B, 512, 1, 1) -> (B, 512)
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        # Global Max Pooling -> (B, 512, 1, 1) -> (B, 512)
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1)

        # Concatenate pooling streams -> (B, 1024)
        img_features = torch.cat([avg_pool, max_pool], dim=1)

        # 3. Late Fusion
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate image features and angle -> (B, 1025)
        fused_features = torch.cat([img_features, angle], dim=1)

        # 4. Classification Head
        logits = self.classifier(fused_features)

        return logits
