import torch
import torch.nn as nn
import timm
from library.config import Config


class ResNetMIL(nn.Module):
    """
    ResNet18-based MIL Network.

    Optimizations:
    - Cite solution_lesson_node_00070: Avoid naive multi-stage feature fusion.
    - Cite solution_lesson_node_00022: Use lighter backbone to enable larger batch size.

    Architecture:
    1. Backbone: ResNet18 (pretrained, num_classes=0 for GAP features).
    2. Context: Conv1d -> LayerNorm -> GELU (Cite solution_lesson_node_00027).
    3. Heads: Linear projection to 8 classes per slice.
    4. Aggregation: Global Max Pooling over slices.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # Initialize ResNet18 with num_classes=0 to get pooled features (B*S, 512)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

        self.feature_dim = self.backbone.num_features

        # 2. Context Module
        # Non-Linear 1D Convolution: Conv1d(k=3, p=1) -> LayerNorm -> GELU
        self.context_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.context_norm = nn.LayerNorm(self.feature_dim)
        self.context_act = nn.GELU()

        # 3. Multi-Task Heads
        self.head = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        b, s, c, h, w = x.shape

        # Merge Batch and Slices dimensions
        x = x.view(b * s, c, h, w)

        # Backbone Forward Pass (includes GAP)
        # Shape: (B*S, 512)
        x = self.backbone(x)

        # Reshape to restore Sequence dimension
        # Shape: (B, S, 512)
        x = x.view(b, s, -1)

        # Context Module
        x = x.permute(0, 2, 1)  # (B, C, S)
        x = self.context_conv(x)
        x = x.permute(0, 2, 1)  # (B, S, C)

        x = self.context_norm(x)
        x = self.context_act(x)

        # Instance-Level Classification
        # Shape: (B, S, Num_Classes)
        logits = self.head(x)

        # Aggregation (Global Max Pooling)
        logits, _ = torch.max(logits, dim=1)

        return logits
