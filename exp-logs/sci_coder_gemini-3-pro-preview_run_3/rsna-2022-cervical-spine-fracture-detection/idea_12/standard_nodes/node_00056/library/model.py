import torch
import torch.nn as nn
import timm
from library.config import Config


class RSNAModel(nn.Module):
    """
    LayerNorm-Native Multi-Task MIL Network.

    Architecture:
    1. Backbone: ConvNeXt-Tiny (LayerNorm-native, pretrained).
    2. Context: 1D Convolution -> LayerNorm -> GELU.
    3. Head: Multi-Task Instance Classifiers (8 classes).
    4. Aggregation: Global Max Pooling over the sequence.
    """

    def __init__(self, pretrained=True):
        super().__init__()

        # 1. Backbone
        # num_classes=0 returns the pooled feature vector (e.g., 768 for tiny)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.IN_CHANS,
            drop_rate=0.0,  # Cite solution_lesson_node_00003
            drop_path_rate=0.0,  # Cite solution_lesson_node_00003
        )

        # Dynamically determine feature dimension
        self.feature_dim = self.backbone.num_features

        # 2. Context Module
        # Structure: Conv1d -> LayerNorm -> GELU
        self.context_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=Config.CONTEXT_KERNEL_SIZE,
            padding=Config.CONTEXT_PADDING,
        )

        self.context_norm = nn.LayerNorm(self.feature_dim)
        self.context_act = nn.GELU()

        # 3. Heads
        # Projects features to 8 distinct logits per slice
        self.head = nn.Linear(self.feature_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq, C, H, W)
                              Default: (B, 64, 3, 224, 224)
        Returns:
            torch.Tensor: Aggregated logits of shape (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # Flatten Batch and Sequence dimensions to process slices in parallel
        x = x.view(b * s, c, h, w)

        # Feature Extraction
        # Output: (B*S, Feature_Dim)
        features = self.backbone(x)

        # Reshape for Context Module: (B, Feature_Dim, S)
        # Conv1d expects (Batch, Channels, Length)
        features = features.view(b, s, self.feature_dim)
        features = features.permute(0, 2, 1)

        # Apply Context (Inter-slice dependencies)
        features = self.context_conv(features)

        # Reshape for LayerNorm and Head: (B, S, Feature_Dim)
        # LayerNorm applies to the last dimension
        features = features.permute(0, 2, 1)
        features = self.context_norm(features)
        features = self.context_act(features)

        # Instance-level Predictions
        # Output: (B, S, Num_Classes)
        instance_logits = self.head(features)

        # 4. Aggregation
        # Global Max Pooling across the sequence dimension
        # Output: (B, Num_Classes)
        pooled_logits, _ = torch.max(instance_logits, dim=1)

        return pooled_logits
