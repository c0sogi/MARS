import torch
import torch.nn as nn
import timm
from library.config import Config


class CatheterModel(nn.Module):
    """
    Catheter and Line Position Detection Model.

    Architecture:
    - Backbone: ConvNeXt-Tiny (Pretrained on ImageNet)
    - Head: Global Average Pooling -> LayerNorm -> Linear (11 classes)

    This architecture is chosen to handle high-resolution inputs (768x768)
    effectively using Layer Normalization to mitigate batch size constraints.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Args:
            pretrained (bool): Whether to load pretrained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(CatheterModel, self).__init__()

        # Load the backbone from timm
        # num_classes=0 and global_pool='' removes the default head and pooling,
        # returning the raw feature maps (B, C, H, W).
        self.backbone = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Get the number of output features from the backbone
        # For convnext_tiny, this is typically 768.
        self.num_features = self.backbone.num_features

        # Custom Classification Head
        # 1. Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 2. Layer Normalization
        # LayerNorm is applied over the channel dimension (C) for each sample.
        # This makes the model robust to small batch sizes compared to BatchNorm.
        self.norm = nn.LayerNorm(self.num_features)

        # 3. Linear Layer
        # Projects features to the 11 target classes.
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, NUM_CLASSES).
        """
        # Backbone feature extraction
        # Output shape: (B, C, H_feat, W_feat)
        features = self.backbone(x)

        # Global Average Pooling
        # Output shape: (B, C, 1, 1)
        x = self.global_pool(features)

        # Flatten
        # Output shape: (B, C)
        x = x.flatten(1)

        # Layer Normalization
        # Output shape: (B, C)
        x = self.norm(x)

        # Classification
        # Output shape: (B, NUM_CLASSES)
        logits = self.fc(x)

        return logits
