import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class IcebergResNet(nn.Module):
    """
    ResNet-18 based model for Iceberg classification with Dual-Stream Pooling
    and Late Fusion of incidence angle.
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load pretrained ResNet-18
        # We use the explicit weights enum as recommended in modern torchvision
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)

        # Extract the feature extractor (layers conv1 through layer4)
        # We discard the original avgpool and fc layers
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

        # ResNet18 layer4 outputs 512 channels
        self.cnn_feature_dim = 512

        # Dual-Stream Pooling: We concat Average Pooling and Max Pooling
        # Output dimension = 512 + 512 = 1024
        self.pooled_dim = self.cnn_feature_dim * 2

        # Late Fusion: We add 1 dimension for the incidence angle
        self.fusion_dim = self.pooled_dim + 1

        # Classification Head
        # Architecture: BatchNorm -> Dropout(0.5) -> Dense -> Logit
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=0.5),
            nn.Linear(self.fusion_dim, 1),
        )

    def forward(self, x, inc_angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, 224, 224).
            inc_angle (torch.Tensor): Incidence angles of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Feature Extraction
        # Output shape: (Batch, 512, 7, 7) for 224x224 input
        features = self.features(x)

        # 2. Dual-Stream Pooling
        # Global Average Pooling -> (Batch, 512)
        avg_pool = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        # Global Max Pooling -> (Batch, 512)
        max_pool = F.adaptive_max_pool2d(features, (1, 1)).view(features.size(0), -1)

        # Concatenate pooled features -> (Batch, 1024)
        pooled_features = torch.cat([avg_pool, max_pool], dim=1)

        # 3. Late Fusion
        # Ensure inc_angle is (Batch, 1)
        inc_angle = inc_angle.view(-1, 1)

        # Concatenate visual features with incidence angle -> (Batch, 1025)
        fused_features = torch.cat([pooled_features, inc_angle], dim=1)

        # 4. Classification
        logits = self.classifier(fused_features)

        return logits
