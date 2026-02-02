import torch
import torch.nn as nn
import torchvision.models as models
from library.configuration import Config


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture with Late Fusion for Incidence Angle.

    Structure:
    1. Backbone: Pretrained ResNet-18 (up to Global Average Pooling).
    2. Fusion: Concatenation of 512-dim image features with 1-dim incidence angle.
    3. Head: BatchNorm -> Dropout -> Linear.
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load Pretrained ResNet18
        # We use the modern weights API supported in torchvision >= 0.13
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Backbone: Keep everything up to the Global Average Pooling layer.
        # ResNet18 structure: conv1->bn1->relu->maxpool->layer1..4->avgpool->fc
        # We remove the final 'fc' layer.
        self.backbone = nn.Sequential(*list(self.resnet.children())[:-1])

        # Feature dimension from ResNet18 GAP is 512
        self.feature_dim = Config.BACKBONE_OUT_DIM

        # Fusion dimension: Image features (512) + Incidence Angle (1)
        fusion_dim = self.feature_dim + 1

        # Minimalist Head
        # As per design: Batch Normalization -> Dropout -> Linear
        # BN here is crucial to normalize the incidence angle relative to the image features
        self.head = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(fusion_dim, 1),
        )

    def forward(self, x, inc_angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image tensor of shape (Batch, 3, H, W).
            inc_angle (torch.Tensor): Incidence angle tensor of shape (Batch, ) or (Batch, 1).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1).
        """
        # 1. Image Feature Extraction
        # Output shape: (Batch, 512, 1, 1)
        features = self.backbone(x)

        # Flatten to (Batch, 512)
        features = features.view(features.size(0), -1)

        # 2. Process Incidence Angle
        # Ensure shape is (Batch, 1)
        inc_angle = inc_angle.view(-1, 1)

        # 3. Late Fusion
        # Concatenate along the feature dimension
        fused = torch.cat((features, inc_angle), dim=1)

        # 4. Classification Head
        out = self.head(fused)

        return out
