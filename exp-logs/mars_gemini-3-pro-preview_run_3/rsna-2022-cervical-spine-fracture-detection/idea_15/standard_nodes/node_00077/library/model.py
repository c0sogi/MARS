import torch
import torch.nn as nn
import timm
from library.config import Config


class ConvNeXtMIL(nn.Module):
    """
    Stabilized 2.5D ConvNeXt Multi-Task MIL Network.

    Architecture:
    1. Backbone: ConvNeXt-Tiny (Pretrained, LayerNorm based).
    2. Context: 1D Convolution over the slice sequence -> LayerNorm -> GELU.
    3. Head: Linear projection to 8 classes per slice.
    4. Aggregation: Global Max Pooling over slices.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_channels=Config.IN_CHANNELS,
    ):
        super().__init__()

        # 1. Backbone
        # num_classes=0 returns the pooled feature vector (B, D)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, in_chans=in_channels
        )
        self.num_features = self.backbone.num_features

        # 2. Context Module
        # Captures Z-axis continuity.
        # Structure: Conv1d -> LayerNorm -> GELU
        # Note: LayerNorm is applied over the feature dimension, so we permute before/after.
        self.context_conv = nn.Conv1d(
            in_channels=self.num_features,
            out_channels=self.num_features,
            kernel_size=3,
            padding=1,
        )
        self.context_norm = nn.LayerNorm(self.num_features)
        self.context_act = nn.GELU()

        # 3. Multi-Task Heads
        # Predicts [C1, C2, C3, C4, C5, C6, C7, Patient_Overall] for each slice
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes)
        """
        b, s, c, h, w = x.shape

        # --- Backbone Feature Extraction ---
        # Collapse Batch and Slices dimensions to process as a standard batch of images
        x = x.view(b * s, c, h, w)

        # Extract features: (B*S, D)
        features = self.backbone(x)

        # Reshape to (B, S, D) to restore sequence structure
        features = features.view(b, s, -1)

        # --- Context Modeling ---
        # Permute to (B, D, S) for Conv1d (Batch, Channels, Length)
        features = features.permute(0, 2, 1)

        # Apply 1D Convolution
        features = self.context_conv(features)

        # Permute back to (B, S, D) for LayerNorm and Linear Head
        features = features.permute(0, 2, 1)

        # Apply Norm and Activation
        features = self.context_norm(features)
        features = self.context_act(features)

        # --- Instance Classification ---
        # Project to logits: (B, S, Num_Classes)
        logits = self.head(features)

        # --- Aggregation ---
        # Global Max Pooling over the slice dimension (dim=1)
        # We want to find the strongest signal for fracture in the entire volume
        # Output shape: (B, Num_Classes)
        pooled_logits, _ = torch.max(logits, dim=1)

        return pooled_logits
