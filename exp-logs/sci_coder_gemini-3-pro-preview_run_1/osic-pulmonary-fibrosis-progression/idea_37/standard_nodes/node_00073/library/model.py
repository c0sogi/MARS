import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Scaled Visual Backbone based on EfficientNet-B1.
    Extracts high-fidelity features by maintaining native dimensionality.
    """

    def __init__(
        self, model_name=Config.BACKBONE_NAME, pretrained=Config.BACKBONE_PRETRAINED
    ):
        super().__init__()
        # efficientnet_b1 output is 1280 dim with global_pool='avg' and num_classes=0
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

    def forward(self, x):
        # Input: (Batch, 3, H, W)
        # Output: (Batch, 1280)
        return self.backbone(x)


class TabularMLP(nn.Module):
    """
    Deep Tabular Alignment module.
    Projects low-dim clinical features to high-dim semantic space via non-linear MLP.
    Structure: Linear -> GeLU -> Linear -> GeLU -> Linear
    """

    def __init__(self, input_dim=7, hidden_dim=Config.TABULAR_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        # Input: (Batch, 7)
        # Output: (Batch, 1280)
        return self.net(x)


class LightweightAttention(nn.Module):
    """
    Lightweight Symmetric Attention Block.
    Consists of Pre-Norm + Multi-Head Self-Attention + Residual.
    Explicitly removes the Feed-Forward Network (FFN) to prevent overfitting.
    """

    def __init__(
        self,
        embed_dim=Config.BACKBONE_DIM,
        num_heads=Config.FUSION_HEADS,
        dropout=Config.FUSION_DROPOUT,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

    def forward(self, x):
        # x: (Batch, SeqLen, EmbedDim)
        residual = x
        x_norm = self.norm(x)
        # Self-Attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        return residual + attn_out


class DALANet(nn.Module):
    """
    Deep-Aligned Lightweight-Attention Network (DALA-Net).
    Fuses dual-view CT scans with clinical metadata to predict FVC decline trajectory.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Scaled Visual Backbones
        # Processing Axial and Coronal views independently
        self.backbone_axial = VisualBackbone()
        self.backbone_coronal = VisualBackbone()

        # 2. Deep Tabular Alignment
        # Input dim is 7 based on data.py encoding
        self.tabular_mlp = TabularMLP(input_dim=7)

        # 3. Lightweight Symmetric Attention (Fusion)
        self.fusion = LightweightAttention()

        # 4. Prior-Anchored Parametric Head
        # Input: Contextualized Visual Residual (1280) + Raw Tabular (7)
        head_in_dim = Config.BACKBONE_DIM + 7

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, Config.HEAD_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(
                Config.HEAD_HIDDEN_DIM, 3
            ),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, axial, coronal, tabular, delta_week, baseline_fvc):
        """
        Args:
            axial: (B, 3, H, W) - Axial view images
            coronal: (B, 3, H, W) - Coronal view images
            tabular: (B, 7) - Normalized clinical features
            delta_week: (B,) - Relative week number (Week - Baseline_Week)
            baseline_fvc: (B,) - Baseline FVC measurement

        Returns:
            fvc_pred: (B, 1) - Predicted FVC
            sigma_pred: (B, 1) - Predicted Confidence
        """
        # Ensure scalar inputs are correctly shaped
        if delta_week.dim() == 1:
            delta_week = delta_week.view(-1, 1)
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.view(-1, 1)

        # -----------------------------------------------------------
        # 1. Feature Extraction
        # -----------------------------------------------------------
        # Extract features maintaining native dimensionality (1280)
        feat_ax = self.backbone_axial(axial)  # (B, 1280)
        feat_cor = self.backbone_coronal(coronal)  # (B, 1280)
        feat_tab = self.tabular_mlp(tabular)  # (B, 1280)

        # -----------------------------------------------------------
        # 2. Fusion (Lightweight Attention)
        # -----------------------------------------------------------
        # Stack tokens: [Axial, Coronal, Tabular]
        tokens = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)  # (B, 3, 1280)

        # Apply attention (Pre-Norm, No FFN)
        tokens = self.fusion(tokens)

        # -----------------------------------------------------------
        # 3. Visual-Exclusive Pooled Readout
        # -----------------------------------------------------------
        # Extract updated visual tokens only (indices 0 and 1)
        v_ax_prime = tokens[:, 0, :]
        v_cor_prime = tokens[:, 1, :]

        # Average Pooling to derive Contextualized Visual Residual
        visual_context = (v_ax_prime + v_cor_prime) / 2.0  # (B, 1280)

        # -----------------------------------------------------------
        # 4. Parametric Inference
        # -----------------------------------------------------------
        # Concatenate Visual Context with RAW Tabular Features (Skip Connection)
        combined = torch.cat([visual_context, tabular], dim=1)  # (B, 1280 + 7)

        # Predict parameters
        params = self.head(combined)  # (B, 3)

        alpha = params[:, 0:1]  # Slope of decline/incline
        sigma_base = F.softplus(params[:, 1:2])  # Base uncertainty (must be > 0)
        sigma_growth = F.softplus(
            params[:, 2:3]
        )  # Uncertainty growth rate (must be > 0)

        # -----------------------------------------------------------
        # 5. Trajectory Calculation
        # -----------------------------------------------------------
        # FVC = Baseline + alpha * delta_t
        fvc_pred = baseline_fvc + alpha * delta_week

        # Sigma = Base + Growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        return fvc_pred, sigma_pred
