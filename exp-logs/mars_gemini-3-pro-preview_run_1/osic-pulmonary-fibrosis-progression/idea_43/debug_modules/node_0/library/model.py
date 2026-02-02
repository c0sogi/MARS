import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DualPathTabularEncoder(nn.Module):
    """
    Splits tabular data into two paths:
    1. Fusion Path: High-dim alignment for attention with visual features.
    2. Prior Path: Low-dim balanced embedding for skip connection.
    """

    def __init__(self, input_dim, fusion_dim, prior_dim, dropout=0.1):
        super().__init__()

        # Path A: Fusion (Deep MLP)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, fusion_dim),
        )

        # Path B: Prior (Shallow MLP)
        self.prior_mlp = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, prior_dim)
        )

    def forward(self, x):
        fusion_emb = self.fusion_mlp(x)
        prior_emb = self.prior_mlp(x)
        return fusion_emb, prior_emb


class BSHDAN(nn.Module):
    """
    Balanced-Skip Holistic Dual-Axis Network.

    Architecture:
    - 2x EfficientNet-B0 (Axial, Coronal) -> 1280 dim
    - Dual-Path Tabular Encoder -> 1280 dim (Fusion) + 128 dim (Prior)
    - Symmetric Self-Attention (Visual + Tabular Fusion)
    - Holistic Pooling (Mean of tokens)
    - Skip Connection (Concat Holistic + Tabular Prior)
    - Parametric Head (Alpha, Sigma_Base, Sigma_Growth)
    """

    def __init__(self):
        super().__init__()

        # 1. Visual Backbones
        # EfficientNet-B0 native output is 1280. num_classes=0 returns the pooled feature vector.
        self.backbone_axial = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        self.backbone_coronal = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # 2. Tabular Encoder
        # Input dim = 7 (Percent, Age, Sex*2, Smoke*3)
        self.tabular_encoder = DualPathTabularEncoder(
            input_dim=7,
            fusion_dim=Config.TABULAR_FUSION_DIM,  # 1280
            prior_dim=Config.TABULAR_PRIOR_DIM,  # 128
            dropout=Config.DROPOUT,
        )

        # 3. Contextualization (Symmetric Attention)
        # Input: Sequence of 3 tokens [Axial, Coronal, Tabular_Fusion]
        # Dim: 1280
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.VISUAL_DIM,
            nhead=Config.ATTN_HEADS,
            dim_feedforward=2048,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm
        )
        self.attention = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.ATTN_LAYERS
        )

        # 4. Prediction Head
        # Input: Holistic (1280) + Prior (128) = 1408
        head_input_dim = Config.VISUAL_DIM + Config.TABULAR_PRIOR_DIM

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(
        self, img_axial, img_coronal, tabular, delta_week=None, baseline_fvc=None
    ):
        """
        Args:
            img_axial: (B, 3, 224, 224)
            img_coronal: (B, 3, 224, 224)
            tabular: (B, 7)
            delta_week: (B,) - Time since baseline (optional, for FVC calculation)
            baseline_fvc: (B,) - Baseline FVC (optional, for FVC calculation)

        Returns:
            dict containing parameters and (if inputs provided) predictions.
        """
        batch_size = img_axial.size(0)

        # --- 1. Feature Extraction ---
        # Visual: (B, 1280)
        feat_ax = self.backbone_axial(img_axial)
        feat_cor = self.backbone_coronal(img_coronal)

        # Tabular: (B, 1280) and (B, 128)
        tab_fusion, tab_prior = self.tabular_encoder(tabular)

        # --- 2. Symmetric Attention ---
        # Stack tokens: (B, 3, 1280)
        tokens = torch.stack([feat_ax, feat_cor, tab_fusion], dim=1)

        # Contextualize
        context_tokens = self.attention(tokens)

        # --- 3. Holistic Readout ---
        # Global Average Pooling across tokens -> (B, 1280)
        holistic_vec = torch.mean(context_tokens, dim=1)

        # --- 4. Balanced Prior Anchoring ---
        # Concatenate with low-dim prior -> (B, 1408)
        combined = torch.cat([holistic_vec, tab_prior], dim=1)

        # --- 5. Parameter Prediction ---
        # Output: (B, 3) -> [alpha, sigma_base_raw, sigma_growth_raw]
        raw_preds = self.head(combined)

        alpha = raw_preds[:, 0]

        # Enforce positivity for sigmas using Softplus
        sigma_base = F.softplus(raw_preds[:, 1])
        sigma_growth = F.softplus(raw_preds[:, 2])

        outputs = {
            "alpha": alpha,
            "sigma_base": sigma_base,
            "sigma_growth": sigma_growth,
        }

        # --- 6. Trajectory Inference (Optional) ---
        if delta_week is not None and baseline_fvc is not None:
            # FVC = Baseline + alpha * delta_week
            fvc_pred = baseline_fvc + alpha * delta_week

            # Confidence = sigma_base + sigma_growth * |delta_week|
            # Note: delta_week can be negative, so we take abs
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            outputs["fvc_pred"] = fvc_pred
            outputs["sigma_pred"] = sigma_pred

        return outputs
