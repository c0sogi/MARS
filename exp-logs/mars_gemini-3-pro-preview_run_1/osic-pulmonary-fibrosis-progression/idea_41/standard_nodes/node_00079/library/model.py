import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Extracts high-fidelity features from CT slices using EfficientNet-B1.
    Output: 1280-dimensional Global Average Pooled vector.
    """

    def __init__(self, pretrained=True):
        super(VisualBackbone, self).__init__()
        # num_classes=0 in timm returns the global pooled features (B, Num_Features)
        # For EfficientNet-B1, this is 1280.
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.output_dim = self.backbone.num_features

    def forward(self, x):
        # x shape: (B, 3, 240, 240)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Projects low-dimensional clinical features into the high-dimensional semantic space
    using a Deep MLP.
    """

    def __init__(self, input_dim, hidden_dim=1280):
        super(TabularEncoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # x shape: (B, input_dim)
        return self.mlp(x)


class FusionLayer(nn.Module):
    """
    Contextualizes visual and tabular tokens using Pre-Norm Symmetric Self-Attention.
    """

    def __init__(self, embed_dim=1280, num_heads=4, dropout=0.2):
        super(FusionLayer, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (B, Seq_Len, Embed_Dim)

        # 1. Pre-Norm Attention with Residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 2. Pre-Norm FFN with Residual
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class ParametricHead(nn.Module):
    """
    Predicts trajectory parameters (alpha, sigma_base, sigma_growth) based on
    fused context and raw clinical priors.
    """

    def __init__(self, context_dim=1280, raw_dim=2):
        super(ParametricHead, self).__init__()

        # Input: Fused Context (1280) + Raw Tabular (2: FVC_base, Percent_base)
        input_dim = context_dim + raw_dim

        self.head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, context, raw_tab):
        # context: (B, 1280)
        # raw_tab: (B, 2)

        # Skip connection: Concatenate fused features with raw priors
        combined = torch.cat([context, raw_tab], dim=1)

        out = self.head(combined)

        # Split outputs
        alpha = out[:, 0]
        sigma_base_raw = out[:, 1]
        sigma_growth_raw = out[:, 2]

        # Apply constraints
        # Alpha is unbounded (slope can be negative or positive, though usually negative for disease)
        # Sigmas must be positive
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        return alpha, sigma_base, sigma_growth


class H2DAN(nn.Module):
    """
    Holistic High-Fidelity Dual-Axis Network.
    Combines independent Axial/Coronal backbones with deep tabular embeddings
    via attention fusion to predict disease progression parameters.
    """

    def __init__(self, tabular_input_dim=7):
        super(H2DAN, self).__init__()

        # 1. Independent Visual Backbones
        self.axial_backbone = VisualBackbone(pretrained=Config.PRETRAINED)
        self.coronal_backbone = VisualBackbone(pretrained=Config.PRETRAINED)

        # 2. Deep Tabular Alignment
        self.tabular_encoder = TabularEncoder(
            input_dim=tabular_input_dim, hidden_dim=Config.TABULAR_HIDDEN_DIM
        )

        # 3. Fusion Layer
        self.fusion = FusionLayer(
            embed_dim=Config.FEATURE_DIM,
            num_heads=Config.ATTENTION_HEADS,
            dropout=Config.DROPOUT_RATE,
        )

        # 4. Parametric Head
        self.head = ParametricHead(
            context_dim=Config.FEATURE_DIM, raw_dim=2  # Baseline FVC, Baseline Percent
        )

    def forward(self, batch):
        """
        Args:
            batch (dict): Dictionary containing:
                - 'axial': (B, 3, 240, 240)
                - 'coronal': (B, 3, 240, 240)
                - 'deep_tab': (B, Tab_Dim) - Normalized features
                - 'raw_tab': (B, 2) - [Baseline_FVC, Baseline_Percent]
                - 'delta_week': (B,) - Time since baseline
        """
        axial_img = batch["axial"]
        coronal_img = batch["coronal"]
        deep_tab = batch["deep_tab"]
        raw_tab = batch["raw_tab"]

        # 1. Feature Extraction
        # (B, 1280)
        v_ax = self.axial_backbone(axial_img)
        v_cor = self.coronal_backbone(coronal_img)

        # (B, 1280)
        v_tab = self.tabular_encoder(deep_tab)

        # 2. Sequence Construction
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # 3. Contextualization (Fusion)
        # (B, 3, 1280)
        context_seq = self.fusion(seq)

        # 4. Holistic Pooling
        # Global Average Pooling across the sequence dimension (tokens)
        # (B, 1280)
        fused_context = torch.mean(context_seq, dim=1)

        # 5. Parameter Prediction
        # alpha: slope of decline/incline
        # sigma_base: uncertainty at t=0
        # sigma_growth: uncertainty growth over time
        alpha, sigma_base, sigma_growth = self.head(fused_context, raw_tab)

        output = {
            "alpha": alpha,
            "sigma_base": sigma_base,
            "sigma_growth": sigma_growth,
        }

        # 6. Trajectory Calculation (if time delta is provided)
        if "delta_week" in batch:
            delta_week = batch["delta_week"]
            baseline_fvc = raw_tab[:, 0]

            # FVC Prediction: Baseline + Slope * Time
            fvc_pred = baseline_fvc + alpha * delta_week

            # Confidence Prediction: Base_Uncertainty + Growth * |Time|
            confidence_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            output["fvc_pred"] = fvc_pred
            output["confidence_pred"] = confidence_pred

        return output
