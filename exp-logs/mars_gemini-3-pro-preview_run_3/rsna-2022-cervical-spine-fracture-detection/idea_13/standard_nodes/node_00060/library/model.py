import torch
import torch.nn as nn
import timm
from library.config import Config


class DualStreamConvNeXt(nn.Module):
    """
    Dual-Resolution ConvNeXt Multi-Task MIL Network.

    Architecture:
    1. Shared Backbone (ConvNeXt-Tiny): Processes Global (resized) and Local (cropped) streams.
    2. Feature Fusion: Concatenates global and local feature vectors.
    3. Context Module: 1D Convolution over the sequence dimension to capture Z-axis continuity.
    4. Multi-Task Heads: Predicts probabilities for C1-C7 and Patient Overall per slice.
    5. Aggregation: Global Max Pooling aggregates slice predictions to exam level.
    """

    def __init__(self):
        super().__init__()

        # 1. Shared Backbone
        # We use num_classes=0 to get the feature vector (pooling is included in the backbone forward)
        # ConvNeXt is LayerNorm-native, ensuring stability with small batches.
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        # Get the feature dimension (e.g., 768 for convnext_tiny)
        self.backbone_dim = self.backbone.num_features

        # 2. Feature Fusion Dimension
        # Concatenation of Global and Local features
        self.fusion_dim = self.backbone_dim * 2

        # 3. Context Module
        # Structure: Conv1d(k=3, p=1) -> LayerNorm -> GELU
        # Reduces dimension from Fusion_Dim to HIDDEN_DIM while mixing sequence info
        self.context_conv = nn.Conv1d(
            in_channels=self.fusion_dim,
            out_channels=Config.HIDDEN_DIM,
            kernel_size=3,
            padding=1,
        )
        self.context_norm = nn.LayerNorm(Config.HIDDEN_DIM)
        self.context_act = nn.GELU()

        # 4. Multi-Task Heads
        # Projects contextualized features to 8 logits (C1-C7, Patient_Overall)
        self.head = nn.Linear(Config.HIDDEN_DIM, Config.NUM_CLASSES)

    def forward(self, global_input, local_input):
        """
        Args:
            global_input (Tensor): (Batch, Slices, 3, H, W) - Resized context.
            local_input (Tensor): (Batch, Slices, 3, H, W) - Center cropped detail.

        Returns:
            Tensor: (Batch, Num_Classes) - Exam-level logits.
        """
        b, s, c, h, w = global_input.shape

        # Combine Batch and Slice dimensions for parallel backbone processing
        # Shape: (B*S, 3, H, W)
        global_flat = global_input.view(b * s, c, h, w)
        local_flat = local_input.view(b * s, c, h, w)

        # Pass through shared backbone
        # Shape: (B*S, Backbone_Dim)
        global_features = self.backbone(global_flat)
        local_features = self.backbone(local_flat)

        # Reshape back to sequence format
        # Shape: (B, S, Backbone_Dim)
        global_features = global_features.view(b, s, -1)
        local_features = local_features.view(b, s, -1)

        # Feature Fusion
        # Shape: (B, S, Backbone_Dim * 2)
        fused_features = torch.cat([global_features, local_features], dim=-1)

        # Context Module
        # Conv1d operates on (Batch, Channels, Length), so we permute S and C
        # Input: (B, Fusion_Dim, S)
        x = fused_features.permute(0, 2, 1)

        # Apply 1D Convolution
        # Output: (B, Hidden_Dim, S)
        x = self.context_conv(x)

        # Permute back for LayerNorm and Linear layers
        # Output: (B, S, Hidden_Dim)
        x = x.permute(0, 2, 1)

        # Apply Norm and Activation
        x = self.context_norm(x)
        x = self.context_act(x)

        # Multi-Task Heads (Per Instance/Slice)
        # Output: (B, S, Num_Classes)
        instance_logits = self.head(x)

        # Aggregation (Global Max Pooling)
        # We take the max logit across all slices for each class
        # Output: (B, Num_Classes)
        exam_logits, _ = torch.max(instance_logits, dim=1)

        return exam_logits
