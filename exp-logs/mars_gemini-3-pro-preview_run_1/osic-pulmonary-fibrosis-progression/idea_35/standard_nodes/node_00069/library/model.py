import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EfficientNetBackbone(nn.Module):
    """
    Extracts high-fidelity visual features using EfficientNet-B0.
    Maintains native dimensionality (1280) to preserve texture signals.
    """

    def __init__(
        self, model_name=Config.BACKBONE_NAME, pretrained=Config.BACKBONE_PRETRAINED
    ):
        super().__init__()
        # num_classes=0 triggers Global Average Pooling and returns the feature vector
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
        )

    def forward(self, x):
        # Input: (B, 3, 224, 224)
        # Output: (B, 1280)
        return self.backbone(x)


class LightweightAttentionBlock(nn.Module):
    """
    A stripped-down Transformer block consisting ONLY of Self-Attention.
    Removes the Feed-Forward Network (FFN) to reduce overfitting on small datasets.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # x: (B, Seq_Len, Embed_Dim)
        residual = x
        x_norm = self.norm(x)

        # Self-Attention: Query=Key=Value=x_norm
        # attn_output shape: (B, Seq_Len, Embed_Dim)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)

        # Residual connection only
        return residual + attn_output


class LARFNet(nn.Module):
    """
    Lightweight-Attention Residual-Fusion Network (LARF-Net).

    Synthesizes independent visual backbones with tabular metadata via
    parameter-efficient attention mechanisms and prior-anchored regression.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent High-Fidelity Visual Backbones
        # Branch A: Axial View
        self.backbone_axial = EfficientNetBackbone()
        # Branch B: Coronal View
        self.backbone_coronal = EfficientNetBackbone()

        # 2. Unified Tabular Projection
        # Projects 7 tabular features UP to 1280 dimensions to match visual features
        self.tabular_projection = nn.Linear(Config.TABULAR_INPUT_DIM, Config.FUSION_DIM)

        # 3. Lightweight Symmetric Attention
        self.attention = LightweightAttentionBlock(
            embed_dim=Config.FUSION_DIM,
            num_heads=Config.ATTN_HEADS,
            dropout=Config.DROPOUT,
        )

        # 4. Prior-Anchored Parametric Head
        # Input: Pooled Context (1280) + Raw Tabular (7) via skip connection
        # We concatenate the raw priors to ensure the head has direct access to them
        head_input_dim = Config.FUSION_DIM + Config.TABULAR_INPUT_DIM

        # Predicts 3 parameters: alpha (slope), sigma_base, sigma_growth
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 64), nn.ReLU(), nn.Linear(64, 3)
        )

    def forward(self, img_axial, img_coronal, tabular, week=None, baseline_fvc=None):
        """
        Args:
            img_axial: (B, 3, 224, 224) - Axial Tri-Slab
            img_coronal: (B, 3, 224, 224) - Coronal Tri-Slab
            tabular: (B, 7) - Normalized tabular features (Age, Percent, Sex, Smoke)
            week: (B,) - Relative week numbers (optional, for trajectory calculation)
            baseline_fvc: (B,) - Baseline FVC values (optional, for trajectory calculation)

        Returns:
            If week/baseline_fvc provided: (fvc_pred, sigma_pred)
            Else: (alpha, sigma_base, sigma_growth)
        """
        # 1. Feature Extraction (Independent processing)
        feat_ax = self.backbone_axial(img_axial)  # (B, 1280)
        feat_cor = self.backbone_coronal(img_coronal)  # (B, 1280)

        # 2. Tabular Projection
        feat_tab = self.tabular_projection(tabular)  # (B, 1280)

        # 3. Stack Sequence for Attention
        # Sequence: [Axial, Coronal, Tabular] -> Shape: (B, 3, 1280)
        seq = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # 4. Lightweight Attention Fusion
        # Contextualizes the modalities without heavy FFN layers
        seq_context = self.attention(seq)  # (B, 3, 1280)

        # 5. Holistic Pooled Readout
        # Global Average Pooling across the sequence dimension
        pooled_context = torch.mean(seq_context, dim=1)  # (B, 1280)

        # 6. Prior-Anchored Skip Connection
        # Concatenate the learned context with the explicit tabular priors
        head_input = torch.cat([pooled_context, tabular], dim=1)  # (B, 1287)

        # 7. Parametric Prediction
        params = self.head(head_input)  # (B, 3)

        alpha = params[:, 0]
        # Enforce positivity for uncertainty estimates
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 8. Trajectory Calculation (if inputs provided)
        if week is not None and baseline_fvc is not None:
            # Linear Decline Model: FVC = Baseline + alpha * relative_week
            fvc_pred = baseline_fvc + alpha * week

            # Uncertainty Model: Sigma = Base + Growth * |relative_week|
            # Note: Clipping to 70 is handled in the loss function/metric, not here
            sigma_pred = sigma_base + sigma_growth * torch.abs(week)

            return fvc_pred, sigma_pred

        return alpha, sigma_base, sigma_growth
