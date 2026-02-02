import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GatedTabularEncoder(nn.Module):
    """
    Projects low-dimensional tabular data into the high-dimensional visual embedding space
    using a Gated Linear Unit (GLU) to prevent noise inflation.

    Args:
        input_dim (int): Dimension of the raw tabular features.
        output_dim (int): Target dimension (matching visual backbone output).
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc_val = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # x: (B, input_dim)
        val = self.fc_val(x)
        gate = torch.sigmoid(self.fc_gate(x))
        # Element-wise multiplication (GLU mechanism)
        return val * gate


class GTVRNet(nn.Module):
    """
    Gated-Tabular Visual-Residual Network (GTVR-Net).

    Architecture:
    1. Independent EfficientNet-B0 backbones for Axial and Coronal views (1280-dim).
    2. Gated Tabular Expansion (6-dim -> 1280-dim).
    3. Symmetric Self-Attention (Visual-Tabular Contextualization).
    4. Visual-Exclusive Pooled Readout (Isolating the visual delta).
    5. Prior-Anchored Parametric Head (Predicting trajectory parameters).
    """

    def __init__(self):
        super().__init__()

        # 1. Independent High-Fidelity Visual Backbones
        # efficientnet_b0 output is 1280-dim when num_classes=0 (global pool)
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        self.feature_dim = Config.FEATURE_DIM  # 1280

        # 2. Gated Tabular Expansion
        # Input features: Age_norm, Sex_enc, Smoke_Ex, Smoke_Never, Smoke_Current, BasePercent_norm (Total 6)
        self.tabular_input_dim = 6
        self.tab_encoder = GatedTabularEncoder(
            input_dim=self.tabular_input_dim, output_dim=self.feature_dim
        )

        # 3. Symmetric Attention
        # Sequence: [Axial, Coronal, Tabular]
        self.norm = nn.LayerNorm(self.feature_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=8,  # 1280 / 8 = 160 dim per head
            batch_first=True,
            dropout=Config.DROPOUT,
        )

        # 4. Parametric Head
        # Inputs: Visual Residual (1280) + Raw Tabular Skip Connection (6)
        head_input_dim = self.feature_dim + self.tabular_input_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 3),  # Outputs: alpha (slope), sigma_base, sigma_growth
        )

    def forward(self, img_axial, img_coronal, tabular, meta):
        """
        Args:
            img_axial (Tensor): (B, 3, 224, 224)
            img_coronal (Tensor): (B, 3, 224, 224)
            tabular (Tensor): (B, 6) Normalized clinical features
            meta (Tensor): (B, 2) [Baseline_FVC, Delta_Week]

        Returns:
            fvc_pred (Tensor): (B, 1) Predicted FVC
            sigma_pred (Tensor): (B, 1) Predicted Confidence
        """
        # --- 1. Feature Extraction ---
        # (B, 1280)
        v_ax = self.backbone_ax(img_axial)
        v_cor = self.backbone_cor(img_coronal)

        # (B, 1280)
        v_tab = self.tab_encoder(tabular)

        # --- 2. Symmetric Attention ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Pre-Normalization
        tokens_norm = self.norm(tokens)

        # Self-Attention (Contextualization)
        attn_out, _ = self.attention(tokens_norm, tokens_norm, tokens_norm)

        # Residual Connection
        tokens = tokens + attn_out

        # --- 3. Visual-Exclusive Readout ---
        # Extract only the updated visual tokens (index 0 and 1)
        # We discard the tabular token (index 2) to ensure the residual is purely visual signal
        v_ax_prime = tokens[:, 0, :]
        v_cor_prime = tokens[:, 1, :]

        # Average Pooling of visual views
        vis_residual = (v_ax_prime + v_cor_prime) / 2.0  # (B, 1280)

        # --- 4. Prior-Anchored Parametric Head ---
        # Skip connection: Concatenate visual residual with raw tabular features
        # Input: (B, 1280 + 6)
        head_input = torch.cat([vis_residual, tabular], dim=1)

        # Predict parameters
        params = self.head(head_input)

        alpha = params[:, 0]
        # Enforce positivity for uncertainty estimates
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # --- 5. Parametric Inference ---
        # Retrieve anchors
        base_fvc = meta[:, 0]  # Baseline FVC
        delta_week = meta[:, 1]  # Week - Baseline_Week

        # Linear Trajectory Model: FVC = Baseline + alpha * time
        fvc_pred = base_fvc + alpha * delta_week

        # Uncertainty Model: Sigma = Base + Growth * |time|
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        return fvc_pred, sigma_pred
