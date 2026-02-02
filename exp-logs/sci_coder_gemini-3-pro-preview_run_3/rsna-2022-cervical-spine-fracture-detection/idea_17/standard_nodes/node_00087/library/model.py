import torch
import torch.nn as nn
import timm
from library.config import Config


class FractureModel(nn.Module):
    """
    Anatomically ROI-Focused ConvNeXt MIL Network.

    Architecture:
    1. Backbone: ConvNeXt-Tiny (Pretrained, Global Average Pooling).
    2. Context: Non-Linear 1D Convolution (Conv1d -> LayerNorm -> GELU).
    3. Heads: Multi-Task Instance Classifiers (Linear projection to 8 classes per slice).
    4. Aggregation: Global Max Pooling across the sequence dimension.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Args:
            backbone_name (str): Name of the timm backbone (default: 'convnext_tiny').
            pretrained (bool): Whether to load ImageNet weights.
            num_classes (int): Number of target classes (default: 8).
        """
        super(FractureModel, self).__init__()

        # 1. Backbone: ConvNeXt-Tiny
        # num_classes=0 and global_pool='avg' ensures we get the feature vector
        # after Global Average Pooling.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.IN_CHANNELS,
        )

        # Determine feature dimension (e.g., 768 for convnext_tiny)
        self.feature_dim = self.backbone.num_features

        # 2. Context Module: Non-Linear 1D Convolution
        # Models Z-axis continuity.
        # Structure: Conv1d(k=3, p=1) -> LayerNorm -> GELU
        self.context_conv = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=3,
            padding=1,
        )
        self.context_norm = nn.LayerNorm(self.feature_dim)
        self.context_act = nn.GELU()

        # 3. Heads: Multi-Task Instance Classifiers
        # Projects features to 8 logits [C1...C7, Patient_Overall] per slice.
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Seq_Len, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (Batch, 8).
        """
        b, s, c, h, w = x.shape

        # --- 1. Backbone Feature Extraction ---
        # Reshape to (B*S, C, H, W) to pass through 2D backbone
        x = x.view(b * s, c, h, w)

        # Extract features: (B*S, feature_dim)
        # Global Average Pooling is handled internally by timm
        features = self.backbone(x)

        # --- 2. Context Modeling ---
        # Reshape to (B, S, feature_dim)
        features = features.view(b, s, self.feature_dim)

        # Permute to (B, feature_dim, S) for Conv1d
        features = features.permute(0, 2, 1)

        # Apply Conv1d
        features = self.context_conv(features)

        # Permute back to (B, S, feature_dim) for LayerNorm and Classifier
        features = features.permute(0, 2, 1)

        # Apply LayerNorm and GELU
        features = self.context_norm(features)
        features = self.context_act(features)

        # --- 3. Instance Classification ---
        # Shape: (B, S, 8)
        logits = self.classifier(features)

        # --- 4. Aggregation ---
        # Global Max Pooling across the sequence dimension (dim=1)
        # Shape: (B, 8)
        pooled_logits, _ = torch.max(logits, dim=1)

        return pooled_logits
