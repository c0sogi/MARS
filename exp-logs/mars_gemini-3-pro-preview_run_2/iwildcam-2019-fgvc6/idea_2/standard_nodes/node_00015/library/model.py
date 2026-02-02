import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from library.config import Config


class HybridResNet(nn.Module):
    """
    Custom ResNet50 architecture with Hybrid Pooling (GAP + GMP).
    Designed for Two-Stage Transfer Learning.
    """

    def __init__(self):
        super(HybridResNet, self).__init__()

        # 1. Load Backbone
        # Use weights=ResNet50_Weights.IMAGENET1K_V1 if PRETRAINED is True
        weights = ResNet50_Weights.IMAGENET1K_V1 if Config.PRETRAINED else None
        original_model = models.resnet50(weights=weights)

        # 2. Extract Feature Extractor
        # Remove the last two layers: AvgPool and FC
        # ResNet50 children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We keep everything up to layer4
        self.backbone = nn.Sequential(*list(original_model.children())[:-2])

        # 3. Define Custom Head
        # ResNet50 output channels at layer4 is 2048.
        # Hybrid pooling concatenates GAP and GMP, so input dim is 2048 * 2 = 4096.
        self.feature_dim = 2048 * 2
        self.fc = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass with Hybrid Pooling.
        """
        # Feature extraction: (B, 3, H, W) -> (B, 2048, H/32, W/32)
        x = self.backbone(x)

        # Hybrid Pooling
        # Global Average Pooling
        x_avg = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        # Global Max Pooling
        x_max = F.adaptive_max_pool2d(x, (1, 1)).flatten(1)

        # Concatenate features
        x_cat = torch.cat([x_avg, x_max], dim=1)

        # Classification
        out = self.fc(x_cat)

        return out

    def freeze_backbone(self):
        """
        Freezes all parameters in the backbone.
        Used for Stage 1 (Linear Warmup).
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Ensure head is trainable
        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze_layer4(self):
        """
        Unfreezes the last residual block (Layer 4) of the backbone.
        Used for Stage 2 (Fine-Tuning).
        """
        # The backbone is a Sequential container.
        # Indices: 0=conv1... 4=layer1, 5=layer2, 6=layer3, 7=layer4

        # First, ensure everything is frozen (good practice before selective unfreeze)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze Layer 4 (index 7 in the sequential list created from children)
        # Note: list(resnet.children())[:-2] results in 8 modules.
        # Index 7 corresponds to the original 'layer4'.
        for param in self.backbone[7].parameters():
            param.requires_grad = True

        # Ensure head is trainable
        for param in self.fc.parameters():
            param.requires_grad = True
