import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalEfficientNet(nn.Module):
    """
    Hierarchical EfficientNet-B4 model.
    Extracts features from the last three reduction stages (strides 8, 16, 32),
    applies Global Average Pooling, concatenates them, and classifies.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load EfficientNet-B4 NS backbone
        # We use features_only=True to extract intermediate feature maps.
        # Indices correspond to strides: 0->2, 1->4, 2->8, 3->16, 4->32.
        # We want strides 8, 16, 32.
        self.backbone = timm.create_model(
            Config.MODEL_EFFNET_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Calculate the total number of channels for the linear layer
        # feature_info.channels() provides the channel count for the selected out_indices
        total_channels = sum(self.backbone.feature_info.channels())

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(total_channels, Config.NUM_CLASSES)

    def forward(self, x):
        # Forward pass through backbone returns a list of feature maps
        features = self.backbone(x)

        # Apply GAP to each feature map and flatten
        pooled_features = []
        for f in features:
            # f shape: [Batch, Channels, Height, Width]
            # After pool: [Batch, Channels, 1, 1]
            # After flatten: [Batch, Channels]
            pooled_features.append(self.pool(f).flatten(1))

        # Concatenate multi-scale features
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification head
        return self.fc(concat_features)


class HierarchicalSwin(nn.Module):
    """
    Hierarchical Swin Transformer (Tiny).
    Extracts output tensors from Stage 2, Stage 3, and Stage 4,
    applies Global Average Pooling, concatenates them, and classifies.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load Swin Tiny backbone
        # Swin stages typically correspond to strides 4, 8, 16, 32.
        # Indices: 0 (Stage 1), 1 (Stage 2), 2 (Stage 3), 3 (Stage 4).
        # We want Stage 2, 3, 4.
        self.backbone = timm.create_model(
            Config.MODEL_SWIN_NAME,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )

        # Calculate total channels
        total_channels = sum(self.backbone.feature_info.channels())

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(total_channels, Config.NUM_CLASSES)

    def forward(self, x):
        # Forward pass through backbone
        features = self.backbone(x)

        # Apply GAP to each feature map and flatten
        # timm ensures features are returned in NCHW format when features_only=True
        pooled_features = []
        for f in features:
            pooled_features.append(self.pool(f).flatten(1))

        # Concatenate multi-scale features
        concat_features = torch.cat(pooled_features, dim=1)

        # Classification head
        return self.fc(concat_features)
