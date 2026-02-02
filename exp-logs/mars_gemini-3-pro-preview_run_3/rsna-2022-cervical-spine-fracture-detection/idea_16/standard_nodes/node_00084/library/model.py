import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SpatialAvgConvNeXt(nn.Module):
    """
    A wrapper around timm's ConvNeXt that uses Global Average Pooling (GAP).
    Cite solution_lesson_node_00079: GAP is preferred over GMP for medical imaging backbones
    to avoid latching onto high-frequency noise artifacts.
    """

    def __init__(self, backbone_name, pretrained=True):
        super().__init__()
        # global_pool='' and num_classes=0 ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )

        # Determine the number of output features dynamically
        # Usually 768 for convnext_tiny
        self.num_features = self.backbone.num_features

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.backbone(x)  # Output: (B, num_features, H/32, W/32)

        # Spatial Average Pooling
        # (B, C, H', W') -> (B, C, 1, 1)
        x = F.adaptive_avg_pool2d(x, (1, 1))

        # Flatten: (B, C, 1, 1) -> (B, C)
        x = x.flatten(1)
        return x


class FractureMILModel(nn.Module):
    """
    ConvNeXt MIL Network with GAP.

    Architecture:
    1. Input: 2.5D Stacks (B, S, 3, H, W)
    2. Backbone: ConvNeXt-Tiny with Spatial Avg Pooling (applied to each slice)
    3. Context: 1D Convolution -> LayerNorm -> GELU (applied over sequence S)
    4. Head: Multi-Task Instance Classifier (8 logits per slice)
    5. Aggregation: Global Max Pooling over sequence S
    """

    def __init__(self, config=Config):
        super().__init__()

        # 1. Backbone
        self.backbone = SpatialAvgConvNeXt(
            backbone_name=config.BACKBONE, pretrained=config.PRETRAINED
        )
        self.dim = self.backbone.num_features
        self.num_classes = config.NUM_CLASSES

        # 2. Context Module
        # Models Z-axis continuity.
        # Structure: Conv1d(k=3, p=1) -> LayerNorm -> GELU
        self.conv1d = nn.Conv1d(
            in_channels=self.dim, out_channels=self.dim, kernel_size=3, padding=1
        )
        self.ln = nn.LayerNorm(self.dim)
        self.act = nn.GELU()

        # 3. Multi-Task Head
        # Projects features to 8 logits: [C1, C2, C3, C4, C5, C6, C7, patient_overall]
        self.head = nn.Linear(self.dim, self.num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Slices, Channels, Height, Width)
               Example: (B, 64, 3, 224, 224)
        Returns:
            logits: Tensor of shape (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # --- Stage 1: Backbone Feature Extraction ---
        # Merge batch and slice dimensions to process in parallel
        x = x.view(b * s, c, h, w)

        # Extract features: (B*S, dim)
        features = self.backbone(x)

        # --- Stage 2: Sequence Modeling (Context) ---
        # Reshape to (B, dim, S) for Conv1d
        # We want to convolve over the slice dimension S
        features = features.view(b, s, self.dim).transpose(1, 2)

        # 1D Convolution
        ctx = self.conv1d(features)  # (B, dim, S)

        # LayerNorm requires channels to be the last dimension
        ctx = ctx.transpose(1, 2)  # (B, S, dim)
        ctx = self.ln(ctx)

        # Activation
        ctx = self.act(ctx)

        # --- Stage 3: Instance Classification ---
        # Predict logits for every slice independently (but with context)
        # ctx: (B, S, dim) -> logits: (B, S, 8)
        instance_logits = self.head(ctx)

        # --- Stage 4: MIL Aggregation ---
        # Global Max Pooling over the slice dimension
        # We take the maximum probability/logit across all slices for each class
        # logits: (B, S, 8) -> (B, 8)
        pooled_logits, _ = torch.max(instance_logits, dim=1)

        return pooled_logits
