import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class IcebergResNet18(nn.Module):
    """
    ResNet-18 based model for Iceberg detection with Late Fusion of incidence angle.

    Architecture:
    1. Backbone: ResNet-18 (Pretrained on ImageNet)
    2. Pooling: Global Average Pooling (GAP)
    3. Fusion: Concatenation of 512-dim image vector and 1-dim angle scalar
    4. Head: BatchNorm -> Dropout -> Linear
    """

    def __init__(self):
        super(IcebergResNet18, self).__init__()

        # Load Pretrained ResNet18
        # We use the modern weights API if available, otherwise fallback (though environment is new)
        if Config.PRETRAINED:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None

        resnet = models.resnet18(weights=weights)

        # Extract feature extractor: Keep everything up to and including the AvgPool layer
        # ResNet children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We remove the last child (fc)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # ResNet18 outputs a 512-dimensional vector after GAP
        self.num_ftrs = 512

        # Fusion Dimension: 512 (Image Features) + 1 (Incidence Angle)
        fusion_dim = self.num_ftrs + 1

        # Minimalist Head definition
        # As per requirements: BatchNorm -> Dropout(0.5) -> Linear
        self.head = nn.Sequential(
            nn.BatchNorm1d(fusion_dim),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(fusion_dim, Config.NUM_CLASSES),
        )

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch_Size, 3, Height, Width)
            angle (torch.Tensor): Normalized incidence angles of shape (Batch_Size,) or (Batch_Size, 1)

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, 1)
        """
        # 1. Image Feature Extraction
        # Input: (B, 3, 224, 224) -> Output: (B, 512, 1, 1)
        x = self.features(x)

        # Flatten to (B, 512)
        x = torch.flatten(x, 1)

        # 2. Angle Processing
        # Ensure angle is (B, 1)
        angle = angle.view(-1, 1)

        # 3. Late Fusion
        # Concatenate image features with the scalar angle
        x = torch.cat((x, angle), dim=1)  # Resulting shape: (B, 513)

        # 4. Classification Head
        out = self.head(x)

        return out
