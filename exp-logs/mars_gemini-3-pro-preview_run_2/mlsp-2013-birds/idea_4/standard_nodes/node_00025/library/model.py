import torch
import torch.nn as nn
from torchvision import models


class ResNet18DualPool(nn.Module):
    """
    ResNet18 with a Dual Pooling Head (Average + Max Pooling).
    Designed for multi-label bird species classification.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object containing model settings
                    (num_classes, pretrained, use_dual_pooling).
        """
        super(ResNet18DualPool, self).__init__()
        self.config = config

        # Load Pretrained ResNet18
        # Using the modern weights API for torchvision >= 0.13
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if config.pretrained else None
        base_model = models.resnet18(weights=weights)

        # Remove the original Global Average Pooling (avgpool) and Fully Connected (fc) layers
        # ResNet18 children: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4, avgpool, fc]
        # We keep everything up to layer4
        self.backbone = nn.Sequential(*list(base_model.children())[:-2])

        # ResNet18 final feature map channels = 512
        self.feature_dim = 512

        # Define Pooling Layers
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Define Classification Head
        # If dual pooling is enabled, input dimension is 512 * 2 = 1024
        if config.use_dual_pooling:
            self.fc = nn.Linear(self.feature_dim * 2, config.num_classes)
        else:
            self.fc = nn.Linear(self.feature_dim, config.num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, H, W)

        Returns:
            torch.Tensor: Logits of shape (Batch, NumClasses)
        """
        # Feature Extraction
        # Output shape: (Batch, 512, H/32, W/32)
        x = self.backbone(x)

        if self.config.use_dual_pooling:
            # Global Average Pooling -> (Batch, 512)
            x_avg = self.avg_pool(x).flatten(1)

            # Global Max Pooling -> (Batch, 512)
            x_max = self.max_pool(x).flatten(1)

            # Concatenate features -> (Batch, 1024)
            x_cat = torch.cat([x_avg, x_max], dim=1)

            # Classification
            logits = self.fc(x_cat)
        else:
            # Standard Global Average Pooling
            x = self.avg_pool(x).flatten(1)
            logits = self.fc(x)

        return logits
