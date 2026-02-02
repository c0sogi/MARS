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
        # Cite debug_lesson_4: Fix Configuration Errors at the Source.
        # features_only=True is not supported for Swin in this timm version,
        # so we load the full model and manually extract features.
        self.backbone = timm.create_model(Config.MODEL_SWIN_NAME, pretrained=pretrained)

        # We want features from stages 1, 2, 3 (0-indexed)
        self.out_indices = (1, 2, 3)

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Cite debug_lesson_6: Determine Feature Dimensions Dynamically via Dummy Forward Pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMG_SIZE_SWIN, Config.IMG_SIZE_SWIN)
            dummy_features = self.extract_features(dummy_input)
            total_channels = dummy_features.shape[1]

        self.fc = nn.Linear(total_channels, Config.NUM_CLASSES)

    def extract_features(self, x):
        # Manual forward pass to extract intermediate features
        x = self.backbone.patch_embed(x)
        if self.backbone.absolute_pos_embed is not None:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        pooled_features = []
        for i, layer in enumerate(self.backbone.layers):
            x = layer(x)
            if i in self.out_indices:
                # x is (B, L, C)
                B, L, C = x.shape
                # Assuming square feature maps: L = H * W
                H = W = int(L**0.5)

                # Reshape to (B, C, H, W) for pooling
                f = x.view(B, H, W, C).permute(0, 3, 1, 2)
                pooled_features.append(self.pool(f).flatten(1))

        # Concatenate multi-scale features
        return torch.cat(pooled_features, dim=1)

    def forward(self, x):
        concat_features = self.extract_features(x)
        return self.fc(concat_features)
