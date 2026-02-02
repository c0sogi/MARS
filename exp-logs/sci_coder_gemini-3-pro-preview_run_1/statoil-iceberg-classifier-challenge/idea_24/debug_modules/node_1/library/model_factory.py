import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library import config


class IcebergResNet(nn.Module):
    """
    ResNet-18 based architecture with Late Fusion for Iceberg Classification.

    Structure:
    1. Backbone: ResNet-18 (pretrained) layers up to the final convolutional block.
    2. Pooling: Global Average Pooling (GAP).
    3. Fusion: Concatenation of image features (512) and incidence angle (1).
    4. Head: BatchNorm1d -> Dropout -> Linear -> Logit.
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load Pretrained ResNet-18
        # Using the modern weights API if available, falling back if necessary
        try:
            weights = (
                models.ResNet18_Weights.IMAGENET1K_V1 if config.PRETRAINED else None
            )
            resnet = models.resnet18(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            resnet = models.resnet18(pretrained=config.PRETRAINED)

        # Remove the fully connected layer (fc) and the average pooling layer (avgpool)
        # We will implement GAP manually to ensure control over dimensions
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Feature dimension for ResNet-18 is 512
        self.num_features = 512

        # Fusion dimension: Image features + 1 Angle feature
        self.fusion_dim = self.num_features + 1

        # Minimalist Head
        # 1. Batch Normalization on the fused vector to stabilize training
        self.bn = nn.BatchNorm1d(self.fusion_dim)

        # 2. Dropout for regularization
        self.dropout = nn.Dropout(p=config.DROPOUT_RATE)

        # 3. Final Linear Layer (Logit output)
        self.fc = nn.Linear(self.fusion_dim, 1)

    def forward(self, x, angle):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Image tensor of shape (B, 3, H, W).
            angle (torch.Tensor): Incidence angle tensor of shape (B,).

        Returns:
            torch.Tensor: Logit of shape (B, 1).
        """
        # 1. Feature Extraction
        # Output shape: (B, 512, H/32, W/32) -> (B, 512, 7, 7) for 224x224 input
        x = self.backbone(x)

        # 2. Global Average Pooling
        # Output shape: (B, 512, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten: (B, 512)
        x = x.view(x.size(0), -1)

        # 3. Late Fusion
        # Ensure angle has shape (B, 1)
        angle = angle.view(-1, 1)

        # Concatenate features
        x = torch.cat([x, angle], dim=1)

        # 4. Classification Head
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def get_model():
    """
    Factory function to instantiate the model and move it to the configured device.
    """
    model = IcebergResNet()
    model = model.to(config.DEVICE)
    return model
