import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone.
    Uses EfficientNet-B0 initialized with ImageNet weights.
    Applies Global Average Pooling (GAP) via num_classes=0 to return native 1280-dim features.
    """

    def __init__(self):
        super().__init__()
        self.model = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,  # Returns pooled features (B, 1280)
        )

    def forward(self, x):
        return self.model(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Processes raw clinical metadata into a robust shared latent vector.
    Structure: Deep MLP (Linear -> GELU -> Linear -> GELU).
    """

    def __init__(self, input_dim=7, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class SLHDAN(nn.Module):
    """
    Shared-Latent Holistic Dual-Axis Network (SLH-DAN).

    Key Components:
    1. Independent Visual Backbones (Axial & Coronal).
    2. Shared Latent Tabular Encoder.
    3. Bifurcated Flow:
       - Alignment Flow: Latent projected to match visual dim for attention.
       - Prior Preservation Flow: Latent passed directly to head.
    4. Pre-Norm Symmetric Attention for holistic context fusion.
    5. Bottleneck Prior-Anchored Head for parametric prediction.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        # 2. Shared Latent Tabular Encoder
        # Input dim is 7 based on data.py (Age, Sex, 3xSmoke, Percent, Baseline_FVC)
        self.tabular_encoder = TabularEncoder(input_dim=7, latent_dim=Config.LATENT_DIM)

        # 3. Alignment Flow Projection
        # Projects 128-dim latent to 1280-dim to match visual tokens
        self.align_layer = nn.Linear(Config.LATENT_DIM, Config.BACKBONE_DIM)

        # 4. Contextualization Phase (Pre-Norm Symmetric Attention)
        # We use a Transformer Encoder Layer with norm_first=True (Pre-Norm)
        # d_model=1280, nhead=4 (1280/4=320 per head), dim_feedforward=2048
        self.context_layer = nn.TransformerEncoderLayer(
            d_model=Config.BACKBONE_DIM,
            nhead=4,
            dim_feedforward=2048,
            dropout=Config.DROPOUT_RATE,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 5. Bottleneck Prior-Anchored Head
        # Input: Concatenation of Holistic Fused Vector (1280) + Shared Latent (128) = 1408
        self.head_bottleneck = nn.Sequential(
            nn.Linear(Config.BACKBONE_DIM + Config.LATENT_DIM, Config.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Output Layer: Predicts alpha (slope), sigma_base, sigma_growth
        self.head_out = nn.Linear(Config.HEAD_HIDDEN_DIM, 3)

    def forward(self, img_ax, img_cor, tabular, time_delta, baseline_fvc):
        """
        Args:
            img_ax: Axial Tri-Slab images (B, 3, 224, 224)
            img_cor: Coronal Tri-Slab images (B, 3, 224, 224)
            tabular: Normalized clinical features (B, 7)
            time_delta: Time difference from baseline (B,)
            baseline_fvc: Baseline FVC value (B,)

        Returns:
            fvc_pred: Predicted FVC (B,)
            sigma_pred: Predicted Confidence (B,)
        """
        # --- Feature Extraction ---
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)
        t_lat = self.tabular_encoder(tabular)  # (B, 128) - Shared Latent

        # --- Bifurcated Flow & Fusion ---
        # Flow A: Alignment for Attention
        t_align = self.align_layer(t_lat)  # (B, 1280)

        # Stack tokens: [Axial, Coronal, Aligned_Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Contextualization (Self-Attention)
        tokens_out = self.context_layer(tokens)  # (B, 3, 1280)

        # Holistic Readout: Global Average Pooling across updated tokens
        h_fused = tokens_out.mean(dim=1)  # (B, 1280)

        # --- Bottleneck Head ---
        # Flow B: Prior Preservation (Concatenate Shared Latent)
        combined = torch.cat([h_fused, t_lat], dim=1)  # (B, 1408)

        # Predict Parameters
        bottleneck = self.head_bottleneck(combined)  # (B, 128)
        preds = self.head_out(bottleneck)  # (B, 3)

        # Extract parameters
        alpha = preds[:, 0]
        sigma_base = F.softplus(preds[:, 1])
        sigma_growth = F.softplus(preds[:, 2])

        # --- Parametric Inference ---
        # FVC = Baseline + alpha * delta_t
        fvc_pred = baseline_fvc + alpha * time_delta

        # Confidence = sigma_base + sigma_growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        return fvc_pred, sigma_pred
