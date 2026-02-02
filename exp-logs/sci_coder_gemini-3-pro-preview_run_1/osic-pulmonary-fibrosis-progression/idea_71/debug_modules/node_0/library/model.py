import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Shared Latent Topology: Deep MLP to project scalars to a robust latent vector.
    Structure: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Intermediate dimension can be somewhat arbitrary, choosing latent_dim/2 or similar
        # Given input is small (4), expanding to 64 then 128 is reasonable.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class DBSLNet(nn.Module):
    """
    Decoupled-Bottleneck Shared-Latent Network (DBSL-Net).

    Key Components:
    1. Dual Independent EfficientNet-B0 Backbones (Axial & Coronal).
    2. Shared Latent Tabular Encoder.
    3. Pre-Norm Symmetric Attention for Contextualization.
    4. Decoupled-Stream Joint Bottleneck.
    5. Balanced Non-Linear Parametric Head.
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Visual Backbones
        # ==========================================
        # Independent backbones for Axial and Coronal views
        # num_classes=0 ensures we get the pooled feature vector (1280-dim for B0)
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME, pretrained=Config.PRETRAINED, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME, pretrained=Config.PRETRAINED, num_classes=0
        )

        # Native backbone dimensionality
        self.vis_dim = Config.BACKBONE_DIM  # 1280

        # ==========================================
        # 2. Shared Latent Tabular Encoder
        # ==========================================
        # Inputs: Age, Percent, Sex, Smoking (4 features)
        self.tabular_input_dim = 4
        self.latent_dim = Config.LATENT_DIM  # 128

        self.tabular_encoder = TabularEncoder(self.tabular_input_dim, self.latent_dim)

        # ==========================================
        # 3. Normalized Bifurcated Flow & Attention
        # ==========================================
        # Flow A: Project Latent (128) to Visual Dim (1280) for alignment
        self.align_proj = nn.Linear(self.latent_dim, self.vis_dim)
        self.align_norm = nn.LayerNorm(self.vis_dim)

        # Contextual Attention
        # Input sequence length: 3 (Axial, Coronal, Tabular)
        # Embed dim: 1280
        self.attention = nn.MultiheadAttention(
            embed_dim=self.vis_dim,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # ==========================================
        # 4. Decoupled-Stream Joint Bottleneck
        # ==========================================
        # Input: Concat(H_vis, H_ctx) -> 1280 + 1280 = 2560
        # Compress to 128 (H_update)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.vis_dim * 2, self.latent_dim),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
        )

        # ==========================================
        # 5. Balanced Non-Linear Head
        # ==========================================
        # Input: Concat(H_update, T_lat) -> 128 + 128 = 256
        # Output: 3 parameters (Alpha, Sigma_Base, Sigma_Growth)
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 128), nn.GELU(), nn.Linear(128, 3)
        )

    def forward(self, img_ax, img_cor, tabular, week, base_week, base_fvc):
        """
        Args:
            img_ax: (B, 3, 224, 224) Axial images
            img_cor: (B, 3, 224, 224) Coronal images
            tabular: (B, 4) Tabular features [Age, Percent, Sex, Smoke]
            week: (B,) Target week
            base_week: (B,) Baseline week
            base_fvc: (B,) Baseline FVC

        Returns:
            fvc_pred: (B,) Predicted FVC
            sigma_pred: (B,) Predicted Confidence
        """
        batch_size = img_ax.size(0)

        # ------------------------------------------
        # 1. Feature Extraction
        # ------------------------------------------
        # Visual Features (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # Tabular Latent (B, 128) -> T_lat
        t_lat = self.tabular_encoder(tabular)

        # ------------------------------------------
        # 2. Contextualization (Attention)
        # ------------------------------------------
        # Flow A: Alignment
        t_align = self.align_proj(t_lat)
        t_align = self.align_norm(t_align)  # (B, 1280)

        # Stack tokens: [V_ax, V_cor, T_align] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Self-Attention
        # attn_output: (B, 3, 1280)
        attn_output, _ = self.attention(tokens, tokens, tokens)

        # Unpack tokens
        v_ax_ctx = attn_output[:, 0, :]
        v_cor_ctx = attn_output[:, 1, :]
        t_align_ctx = attn_output[:, 2, :]

        # ------------------------------------------
        # 3. Decoupled Aggregation & Bottleneck
        # ------------------------------------------
        # Visual Context: Mean(V'_ax, V'_cor) -> (B, 1280)
        h_vis = (v_ax_ctx + v_cor_ctx) / 2.0

        # Tabular Context: T'_align -> (B, 1280)
        h_ctx = t_align_ctx

        # Joint Compression
        # Concat -> (B, 2560)
        joint_feat = torch.cat([h_vis, h_ctx], dim=1)

        # Update Vector -> (B, 128)
        h_update = self.bottleneck(joint_feat)

        # ------------------------------------------
        # 4. Parametric Prediction
        # ------------------------------------------
        # Assembly: Concat(H_update, T_lat) -> (B, 256)
        # This enforces 50/50 balance between the visual update and the raw prior
        final_feat = torch.cat([h_update, t_lat], dim=1)

        # Head prediction -> (B, 3)
        # [Alpha, Sigma_Base, Sigma_Growth]
        params = self.head(final_feat)

        alpha = params[:, 0]
        sigma_base_raw = params[:, 1]
        sigma_growth_raw = params[:, 2]

        # Apply activations
        # Alpha is linear (can be negative slope)
        # Sigmas must be positive -> Softplus
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        # ------------------------------------------
        # 5. Anchored Trajectory Logic
        # ------------------------------------------
        # Delta t
        delta_week = week - base_week

        # FVC Prediction
        # FVC = Baseline + Alpha * (Week - Baseline_Week)
        fvc_pred = base_fvc + alpha * delta_week

        # Confidence Prediction
        # Sigma = Sigma_Base + Sigma_Growth * |Week - Baseline_Week|
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        return fvc_pred, sigma_pred
