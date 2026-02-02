import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based architecture for Iceberg vs Ship classification.

    Attributes:
        backbone (nn.Sequential): The pretrained ResNet-18 feature extractor (up to avgpool).
        head (nn.Sequential): The classification head with Batch Normalization, Dropout, and Linear layer.
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet-18
        # We strictly adhere to the finding that shallow, dense networks (ResNet-18) work best.
        # Input is 3 channels (Band 1, Band 2, Avg), so standard conv1 is compatible.
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        resnet = models.resnet18(weights=weights)

        # Extract layers: conv1 through avgpool (exclude the final fc layer)
        # ResNet structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet-18 Global Average Pooling output dimension is 512
        self.num_features = 512

        # Late Fusion Dimension: 512 features + 1 scalar (incidence angle)
        self.fusion_dim = self.num_features + 1

        # Minimalist Head
        # Structure: Batch Normalization -> Dropout (0.5) -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.fusion_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (Batch, 3, Height, Width).
            angle (torch.Tensor): Normalized incidence angle tensor of shape (Batch,).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Feature Extraction
        # Pass through ResNet backbone
        x = self.backbone(x)  # Shape: (Batch, 512, 1, 1)

        # Flatten for dense layers
        x = torch.flatten(x, 1)  # Shape: (Batch, 512)

        # 2. Late Fusion
        # Reshape angle to (Batch, 1) for concatenation
        angle = angle.view(-1, 1)

        # Concatenate image features with scalar angle
        x = torch.cat([x, angle], dim=1)  # Shape: (Batch, 513)

        # 3. Classification Head
        logits = self.head(x)  # Shape: (Batch, 1)

        return logits
