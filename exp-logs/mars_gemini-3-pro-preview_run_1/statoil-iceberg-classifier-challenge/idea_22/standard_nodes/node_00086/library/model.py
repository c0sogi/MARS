import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class IcebergResNet(nn.Module):
    """
    ResNet-18 with Late Fusion (Concatenation) of Incidence Angle.
    Cite solution_lesson_node_00082: Prefer Concatenation over Gating/Multiplication.
    Cite solution_lesson_node_00024: Minimalist Classification Head.
    """

    def __init__(self):
        super(IcebergResNet, self).__init__()

        # Load Pretrained ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        base_model = models.resnet18(weights=weights)

        # Extract feature extractor (conv1 up to layer4)
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
        )

        # Minimalist Classification Head
        # Input: 512 (Image Features) + 1 (Angle) = 513
        self.head = nn.Sequential(
            nn.BatchNorm1d(Config.FEATURE_DIM + 1),
            nn.Dropout(p=Config.DROPOUT_RATE),
            nn.Linear(Config.FEATURE_DIM + 1, 1),
        )

    def forward(self, images, angles):
        # 1. Extract Visual Features
        x = self.features(images)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)  # (B, 512)

        # 2. Concatenate Angle
        if angles.dim() == 1:
            angles = angles.view(-1, 1)

        # Cite solution_lesson_node_00082: Concatenation (Late Fusion)
        x = torch.cat([x, angles], dim=1)  # (B, 513)

        # 3. Classification Head
        logits = self.head(x)

        return logits


# Cite debug_lesson_14: Preserve Backward Compatibility with Class Aliases
AngleGatedResNet = IcebergResNet
