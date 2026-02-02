import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.

    Formula: f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN gradients with pow
        x = x.clamp(min=self.eps)
        # Average pooling on x^p
        x_pow = x.pow(self.p)
        # Global Average Pooling over spatial dimensions (H, W)
        avg_x_pow = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # Raise to power 1/p
        gem_out = avg_x_pow.pow(1.0 / self.p)
        return gem_out


class AppleMultiTaskModel(nn.Module):
    """
    Heterogeneous Multi-Task Model for Apple Disease Detection.

    Features:
    - Backbone: Supports EfficientNetV2 and MaxViT via timm.
    - Multi-Level Features: Extracts features from strides 8, 16, and 32.
    - Aggregation: GeM pooling on each level, followed by concatenation.
    - Heads: Decoupled heads for Main task (4-class) and Aux tasks (Binary).
    """

    def __init__(
        self, backbone_name, num_classes=4, pretrained=True, gem_p=3.0, dropout=0.2
    ):
        super(AppleMultiTaskModel, self).__init__()

        # 1. Backbone
        # features_only=True returns a list of feature maps
        # out_indices=(2, 3, 4) corresponds to strides 8, 16, 32 for most standard backbones
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # 2. Determine Feature Dimensions
        # We pass a dummy input to get the channel counts for the selected stages
        # This is robust against different backbones having different channel widths
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)

        # features is a list of tensors
        feature_channels = [f.shape[1] for f in features]
        total_embedding_size = sum(feature_channels)

        # 3. Pooling Layers
        # We use a shared GeM layer logic, or independent parameters?
        # Strategy says "apply GeM... to each stage independently".
        # We'll use a single GeM module instance which learns a scalar p,
        # applied to all maps. (Alternatively, one could use separate GeMs per stage
        # to learn different p values, but a single global p is standard and robust).
        self.gem = GeM(p=gem_p)

        # 4. Regularization
        self.dropout = nn.Dropout(p=dropout)

        # 5. Decoupled Heads
        # Main Head: 4 classes (Healthy, Multiple, Rust, Scab)
        self.head_main = nn.Linear(total_embedding_size, num_classes)

        # Aux Head A: Rust (Binary)
        self.head_aux_rust = nn.Linear(total_embedding_size, 1)

        # Aux Head B: Scab (Binary)
        self.head_aux_scab = nn.Linear(total_embedding_size, 1)

        # Aux Head C: Healthy (Binary)
        self.head_aux_healthy = nn.Linear(total_embedding_size, 1)

        print(f"Initialized {backbone_name} with Multi-Level GeM Features.")
        print(
            f"Feature Channels: {feature_channels} -> Aggregated Size: {total_embedding_size}"
        )

    def forward(self, x):
        # Extract features
        # Returns list of [feat_stride_8, feat_stride_16, feat_stride_32]
        features = self.backbone(x)

        pooled_features = []
        for f in features:
            # Apply GeM pooling: (B, C, H, W) -> (B, C, 1, 1)
            gem_feat = self.gem(f)
            # Flatten: (B, C, 1, 1) -> (B, C)
            flat_feat = torch.flatten(gem_feat, 1)
            pooled_features.append(flat_feat)

        # Concatenate features from all levels
        # Shape: (B, Sum_Channels)
        global_embedding = torch.cat(pooled_features, dim=1)

        # Apply Dropout
        embedding = self.dropout(global_embedding)

        # Forward through heads
        # Note: We return raw logits. Softmax/Sigmoid is applied in Loss function or Inference.
        out_main = self.head_main(embedding)
        out_rust = self.head_aux_rust(embedding)
        out_scab = self.head_aux_scab(embedding)
        out_healthy = self.head_aux_healthy(embedding)

        return {
            "main": out_main,
            "aux_rust": out_rust,
            "aux_scab": out_scab,
            "aux_healthy": out_healthy,
        }
