import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture with Global Average Pooling (GAP) and Late Fusion for incidence angle.
    Using GAP instead of GeM/Max Pooling as it is more robust to speckle noise (Cite 00019).
    """

    def __init__(self, pretrained=True, dropout_rate=Config.DROPOUT_RATE):
        """
        Args:
            pretrained (bool): Whether to load ImageNet weights.
            dropout_rate (float): Dropout probability in the head.
        """
        super(IcebergResNet, self).__init__()

        # Load ResNet18 backbone
        if pretrained:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        base_model = models.resnet18(weights=weights)

        # Remove the original Average Pooling and FC layer
        # ResNet18 structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We take everything before avgpool to keep spatial dimensions (N, 512, 7, 7)
        self.features = nn.Sequential(*list(base_model.children())[:-2])

        # ResNet18 output channels are 512
        self.num_features = 512

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        # Inputs: 512 (Image Features) + 1 (Incidence Angle)
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features + 1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.num_features + 1, 1),
        )

    def forward(self, x, angle):
        """
        Args:
            x (torch.Tensor): Image tensor (Batch, 3, 224, 224)
            angle (torch.Tensor): Incidence angle tensor (Batch,) or (Batch, 1)

        Returns:
            torch.Tensor: Logits (Batch, 1)
        """
        # 1. Backbone Feature Extraction
        x = self.features(x)  # Output: (Batch, 512, 7, 7)

        # 2. Global Average Pooling
        x = self.pool(x)  # Output: (Batch, 512, 1, 1)

        # 3. Flatten
        x = x.view(x.size(0), -1)  # Output: (Batch, 512)

        # 4. Late Fusion
        # Ensure angle is (Batch, 1)
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate features and angle
        x = torch.cat([x, angle], dim=1)  # Output: (Batch, 513)

        # 5. Classification Head
        x = self.head(x)  # Output: (Batch, 1)

        return x
