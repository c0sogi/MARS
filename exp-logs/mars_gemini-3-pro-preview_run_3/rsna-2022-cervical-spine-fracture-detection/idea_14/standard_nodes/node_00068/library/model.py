import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiStageConvNeXtMIL(nn.Module):
    """
    Multi-Stage Feature-Fused ConvNeXt MIL Network.

    Architecture:
    1. Backbone: ConvNeXt-Tiny (pretrained, features_only).
    2. Multi-Stage Fusion: GAP on last 3 stages -> Concatenate.
    3. Context: Conv1d -> LayerNorm -> GELU.
    4. Heads: Linear projection to 8 classes per slice.
    5. Aggregation: Global Max Pooling over slices.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Initialize ConvNeXt-Tiny with features_only=True to access intermediate stages
        # We use in_chans=3 for the 2.5D input (z-1, z, z+1)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimensions dynamically
        # We use the last 3 stages (indices -3, -2, -1 which usually correspond to strides 8, 16, 32)
        feature_info = self.backbone.feature_info.info
        self.selected_indices = [-3, -2, -1]

        total_features = 0
        for i in self.selected_indices:
            total_features += feature_info[i]["num_chs"]

        self.fusion_dim = total_features

        # 2. Context Module
        # Non-Linear 1D Convolution: Conv1d(k=3, p=1) -> LayerNorm -> GELU
        # Captures Z-axis dependencies
        self.context_conv = nn.Conv1d(
            in_channels=self.fusion_dim,
            out_channels=self.fusion_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.context_norm = nn.LayerNorm(self.fusion_dim)
        self.context_act = nn.GELU()

        # 3. Multi-Task Heads
        # Project fused features to 8 logits (C1-C7, Patient Overall)
        self.head = nn.Linear(self.fusion_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Slices, Channels, H, W)
        Returns:
            logits: Tensor of shape (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # Merge Batch and Slices dimensions for backbone processing
        # (B, S, C, H, W) -> (B*S, C, H, W)
        x = x.view(b * s, c, h, w)

        # Backbone Forward Pass
        # Returns a list of feature maps
        features = self.backbone(x)

        # Multi-Stage Feature Fusion
        pooled_features = []
        for i in self.selected_indices:
            f = features[i]  # Shape: (B*S, C_i, H_i, W_i)
            # Global Average Pooling: (B*S, C_i, H_i, W_i) -> (B*S, C_i)
            f = f.mean(dim=[-2, -1])
            pooled_features.append(f)

        # Concatenate features from different stages
        # Shape: (B*S, Total_C)
        x = torch.cat(pooled_features, dim=1)

        # Reshape to restore Sequence dimension for Context Module
        # Shape: (B, S, Total_C)
        x = x.view(b, s, -1)

        # Context Module
        # Conv1d expects (Batch, Channels, Length)
        x = x.permute(0, 2, 1)  # (B, Total_C, S)
        x = self.context_conv(x)
        x = x.permute(0, 2, 1)  # (B, S, Total_C)

        # LayerNorm and Activation
        x = self.context_norm(x)
        x = self.context_act(x)

        # Instance-Level Classification
        # Shape: (B, S, Num_Classes)
        logits = self.head(x)

        # Aggregation (Global Max Pooling)
        # Max pool across the slice dimension (dim=1)
        # Shape: (B, Num_Classes)
        logits, _ = torch.max(logits, dim=1)

        return logits
