import torch
import torch.nn as nn
import torchvision.models as models


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Ship vs Iceberg classification.

    Architecture:
    1. Backbone: Pretrained ResNet-18 (up to Global Average Pooling).
    2. Fusion: Late fusion of visual features (512-d) and incidence angle (1-d).
    3. Head: Minimalist head (BatchNorm -> Dropout -> Linear).
    """

    def __init__(self, pretrained: bool = True, dropout_rate: float = 0.5):
        """
        Initialize the model.

        Args:
            pretrained (bool): Whether to load ImageNet pretrained weights.
            dropout_rate (float): Dropout probability for the classification head.
        """
        super(IcebergResNet18, self).__init__()

        # Load ResNet-18 backbone
        # Use 'weights' parameter for newer torchvision versions, fallback to 'pretrained'
        if hasattr(models, "ResNet18_Weights") and pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            self.backbone = models.resnet18(pretrained=pretrained)

        # Remove the final fully connected layer (fc)
        # ResNet-18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1-4 -> avgpool -> fc
        # We keep everything up to avgpool. The avgpool layer in ResNet is an
        # AdaptiveAvgPool2d((1, 1)), which handles the 224x224 input size correctly.
        layers = list(self.backbone.children())[:-1]
        self.features = nn.Sequential(*layers)

        # ResNet-18 feature dimension is 512
        self.num_features = 512

        # Fusion dimension: 512 (image features) + 1 (incidence angle)
        self.fusion_dim = self.num_features + 1

        # Minimalist Classification Head
        # As per design: BatchNorm -> Dropout -> Linear Probe
        self.head = nn.Sequential(
            nn.BatchNorm1d(self.fusion_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.fusion_dim, 1),
        )

    def forward(self, x, angle):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images, shape (Batch, 3, H, W).
            angle (torch.Tensor): Incidence angles, shape (Batch,) or (Batch, 1).

        Returns:
            torch.Tensor: Logits, shape (Batch, 1).
        """
        # 1. Feature Extraction
        # x: (B, 3, 224, 224) -> features: (B, 512, 1, 1)
        f = self.features(x)
        f = torch.flatten(f, 1)  # (B, 512)

        # 2. Angle Processing
        # Ensure angle is (B, 1) to match feature dimensions for concatenation
        if angle.dim() == 1:
            angle = angle.unsqueeze(1)

        # 3. Late Fusion
        # Concatenate image features and angle along the feature dimension
        fused = torch.cat((f, angle), dim=1)  # (B, 513)

        # 4. Classification Head
        out = self.head(fused)  # (B, 1)

        return out
