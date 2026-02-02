import torch
import torch.nn as nn
import timm
from library.config import Config


class DynamicDepthConvNeXt(nn.Module):
    """
    2.5D ConvNeXt-Tiny with Non-Linear Context and Multi-Task Heads.

    Implements the architecture defined in the design idea:
    1. Backbone: ConvNeXt-Tiny (2D, Pretrained, GAP) processing 2.5D stacks.
    2. Context: 1D Conv -> LayerNorm -> GELU (Inter-slice dependencies).
    3. Heads: Linear projection to 8 logits per slice (Multi-Task).
    4. Aggregation: Global Max Pooling across sequence (MIL).
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ):
        super().__init__()

        # --- 1. Backbone ---
        # ConvNeXt-Tiny natively employs Layer Normalization.
        # We use Global Average Pooling (GAP) for spatial reduction.
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default classification head
            in_chans=Config.IN_CHANNELS,  # 3 channels for 2.5D input
            global_pool="avg",  # Spatial pooling
        )

        # Dynamically determine feature dimension (usually 768 for tiny)
        with torch.no_grad():
            dummy = torch.zeros(1, Config.IN_CHANNELS, 224, 224)
            features = self.encoder(dummy)
            self.feature_dim = features.shape[1]

        # --- 2. Context Module ---
        # Non-Linear 1D Convolution: Conv1d -> LayerNorm -> GELU
        # Captures Z-axis continuity without complex residuals.
        self.context_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.context_norm = nn.LayerNorm(self.feature_dim)
        self.context_act = nn.GELU()

        # --- 3. Multi-Task Heads ---
        # Projects contextualized features to 8 distinct logits per slice.
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input batch of shape (Batch, Seq, Channels, Height, Width).
                              e.g., (B, 64, 3, 224, 224).
        Returns:
            torch.Tensor: Aggregated logits of shape (Batch, Num_Classes).
        """
        b, s, c, h, w = x.shape

        # Fold Batch and Sequence dimensions to process all slices in parallel
        # Shape: (B*S, C, H, W)
        x = x.view(b * s, c, h, w)

        # Extract features using 2D Backbone
        # Shape: (B*S, Feature_Dim)
        features = self.encoder(x)

        # Reshape for Context Module
        # Conv1d expects (Batch, Channels, Length) -> (B, Feature_Dim, S)
        features = features.view(b, s, self.feature_dim).permute(0, 2, 1)

        # Apply Context Module (Z-axis processing)
        features = self.context_conv(features)

        # Permute back for LayerNorm and Linear Head
        # Shape: (B, S, Feature_Dim)
        features = features.permute(0, 2, 1)
        features = self.context_norm(features)
        features = self.context_act(features)

        # Project to Instance Logits
        # Shape: (B, S, Num_Classes)
        instance_logits = self.head(features)

        # --- 4. Aggregation ---
        # Global Max Pooling across the sequence dimension.
        # This assumes that if a fracture exists in ANY slice, the logit for that slice
        # will be high, driving the patient-level prediction.
        pooled_logits, _ = torch.max(instance_logits, dim=1)  # (B, Num_Classes)

        return pooled_logits
