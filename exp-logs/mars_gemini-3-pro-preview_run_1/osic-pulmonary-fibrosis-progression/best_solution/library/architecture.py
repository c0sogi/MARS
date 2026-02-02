import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionFusedDualAxisNet(nn.Module):
    """
    Attention-Fused Dual-Axis Network for Lung Function Decline Prediction.

    Architecture:
    1. Two independent EfficientNet-B0 backbones for Axial and Coronal Tri-Slab MIPs.
    2. An MLP for tabular clinical features.
    3. A Multi-Head Self-Attention module to fuse visual and clinical features.
    4. A parametric regression head predicting slope (alpha) and uncertainty (sigma).
    """

    def __init__(
        self, tabular_input_dim=6, feature_dim=1280, num_heads=4, pretrained=True
    ):
        """
        Args:
            tabular_input_dim (int): Number of input tabular features (default 6).
            feature_dim (int): Dimension of the feature space (default 1280 for EfficientNet-B0).
            num_heads (int): Number of attention heads for fusion.
            pretrained (bool): Whether to load ImageNet weights for backbones.
        """
        super(AttentionFusedDualAxisNet, self).__init__()

        # 1. Independent Visual Backbones
        # EfficientNet-B0 outputs 1280-dim features when num_classes=0
        # Branch A: Axial View
        self.backbone_axial = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )

        # Branch B: Coronal View
        self.backbone_coronal = timm.create_model(
            "efficientnet_b0", pretrained=pretrained, num_classes=0
        )

        # 2. Tabular Embedding MLP
        # Split into embedding and projection to allow skip connection
        self.tab_embedding = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
        )

        self.tab_projection = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        # 3. Attention-Based Feature Fusion
        # Input sequence length: 3 (Axial, Coronal, Tabular)
        # Embedding dimension: 1280
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )

        # 4. Parametric Prediction Head
        # Predicts: alpha (slope), sigma_base, sigma_growth
        # Input: Context Vector (1280) + Tabular Skip (128)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 128, 512), nn.ReLU(), nn.Linear(512, 3)
        )

    def forward(self, images, tabular):
        """
        Args:
            images: Tensor of shape (B, 2, 3, 224, 224)
                    images[:, 0] is Axial View
                    images[:, 1] is Coronal View
            tabular: Tensor of shape (B, 6)

        Returns:
            alpha: Slope of decline (B,)
            sigma_base: Base confidence (B,)
            sigma_growth: Uncertainty growth over time (B,)
        """
        batch_size = images.size(0)

        # --- 1. Feature Extraction ---

        # Extract Axial Features
        # Input: (B, 3, 224, 224) -> Output: (B, 1280)
        img_axial = images[:, 0]
        feat_axial = self.backbone_axial(img_axial)

        # Extract Coronal Features
        # Input: (B, 3, 224, 224) -> Output: (B, 1280)
        img_coronal = images[:, 1]
        feat_coronal = self.backbone_coronal(img_coronal)

        # Extract Tabular Features
        # Input: (B, 6) -> Output: (B, 128) -> (B, 1280)
        tab_embed = self.tab_embedding(tabular)
        feat_tab = self.tab_projection(tab_embed)

        # --- 2. Feature Fusion (Attention) ---

        # Stack features into a sequence: [Axial, Coronal, Tabular]
        # Shape: (B, 3, 1280)
        seq = torch.stack([feat_axial, feat_coronal, feat_tab], dim=1)

        # Apply Self-Attention
        # attn_output shape: (B, 3, 1280)
        # We use the same sequence for Query, Key, and Value (Self-Attention)
        attn_output, _ = self.attention(seq, seq, seq)

        # Global Pooling (Average over the sequence dimension)
        # Shape: (B, 1280)
        context_vector = torch.mean(attn_output, dim=1)

        # Concatenate Tabular Skip Connection
        # Shape: (B, 1280 + 128)
        combined_features = torch.cat([context_vector, tab_embed], dim=1)

        # --- 3. Prediction ---

        # Shape: (B, 3)
        preds = self.head(combined_features)

        # Split outputs
        alpha = preds[:, 0]

        # Enforce positivity for sigma values using Softplus
        # Sigma must be positive for valid metric calculation
        sigma_base = F.softplus(preds[:, 1])
        sigma_growth = F.softplus(preds[:, 2])

        return alpha, sigma_base, sigma_growth
