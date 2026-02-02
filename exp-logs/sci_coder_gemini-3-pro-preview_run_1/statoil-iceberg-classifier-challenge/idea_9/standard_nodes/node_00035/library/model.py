import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Iceberg vs Ship classification.
    Implements Late Fusion of incidence angle and a minimalist classification head.
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load pretrained ResNet18
        # Using 'DEFAULT' weights which corresponds to the best available ImageNet weights
        weights = models.ResNet18_Weights.DEFAULT
        self.backbone = models.resnet18(weights=weights)

        # Extract feature extractor (everything before avgpool and fc)
        # ResNet18 structure: conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc
        # We use layers up to layer4 to get the feature maps
        self.features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )

        # Global Average Pooling
        # Output of layer4 is [B, 512, H/32, W/32] (e.g., 7x7 for 224x224 input)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Feature dimension from ResNet18
        self.num_features = 512

        # Late Fusion Head
        # Concatenates 512 image features + 1 angle feature
        # Structure: BatchNorm -> Dropout -> Linear
        # The BatchNorm here serves to normalize the incidence angle relative to the image features
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.num_features + 1),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(self.num_features + 1, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image tensor of shape [B, 3, H, W]
            angle (torch.Tensor): Incidence angle tensor of shape [B] or [B, 1]

        Returns:
            torch.Tensor: Logits of shape [B, 1]
        """
        # Feature extraction
        x = self.features(x)  # [B, 512, 7, 7]
        x = self.gap(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]

        # Process angle
        # Ensure angle is [B, 1]
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        # Concatenate features and angle
        # x: [B, 512], angle: [B, 1] -> fused: [B, 513]
        fused = torch.cat([x, angle], dim=1)

        # Classification head
        out = self.head(fused)  # [B, 1]

        return out
