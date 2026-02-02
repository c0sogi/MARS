import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes static tabular features into a Shared Latent Vector (T_lat).
    Architecture: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim=4, output_dim=128):
        super(TabularEncoder, self).__init__()
        # Intermediate dimension can be slightly expanded or kept same.
        # Given small input (4), expanding to 64 is reasonable before 128.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, output_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class SCSLNet(nn.Module):
    """
    Split-Context Shared-Latent Network (SCSL-Net).

    Features:
    - Dual Independent Visual Backbones (Axial/Coronal)
    - Shared Latent Tabular Encoder
    - Pre-Norm Symmetric Attention Fusion
    - Split-Stream Compressed Readout (50% Prior, 50% Context)
    - Parametric Output (alpha, sigma_base, sigma_growth)
    """

    def __init__(self):
        super(SCSLNet, self).__init__()

        # 1. Independent Visual Backbones
        # EfficientNet-B0 output is 1280 dim
        self.backbone_ax = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # 2. Tabular Encoder (Shared Latent)
        # Input: Age, Sex, Smoking, Percent (4 dims)
        self.tabular_encoder = TabularEncoder(
            input_dim=len(Config.TABULAR_COLS), output_dim=Config.LATENT_DIM
        )

        # 3. Alignment Layer
        # Projects T_lat (128) to T_align (1280) + LayerNorm
        self.align_proj = nn.Linear(Config.LATENT_DIM, Config.ALIGN_DIM)
        self.align_norm = nn.LayerNorm(Config.ALIGN_DIM)

        # 4. Fusion (Pre-Norm Transformer Encoder Layer)
        # Sequence length: 3 (Axial, Coronal, Tabular)
        # d_model: 1280
        self.fusion = nn.TransformerEncoderLayer(
            d_model=Config.ALIGN_DIM,
            nhead=8,  # 1280 / 8 = 160 per head
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm
        )
        # We use a single layer as per design
        self.transformer = nn.TransformerEncoder(self.fusion, num_layers=1)

        # 5. Split Readout Projections
        # Visual Context Stream Projection (1280 -> 64)
        self.vis_stream_proj = nn.Linear(Config.ALIGN_DIM, Config.CONTEXT_DIM)

        # Tabular Context Stream Projection (1280 -> 64)
        self.ctx_stream_proj = nn.Linear(Config.ALIGN_DIM, Config.CONTEXT_DIM)

        # 6. Parametric Head
        # Input: 64 (Vis) + 64 (Ctx) + 128 (Prior) = 256
        combined_dim = Config.CONTEXT_DIM + Config.CONTEXT_DIM + Config.LATENT_DIM
        self.head = nn.Linear(combined_dim, 3)

        # Initialize weights for head to small values to prevent initial instability
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, img_axial, img_coronal, tabular):
        """
        Args:
            img_axial: (B, 3, 224, 224)
            img_coronal: (B, 3, 224, 224)
            tabular: (B, 4) [Age, Sex, Smoke, Percent]

        Returns:
            preds: (B, 3) -> [alpha, sigma_base, sigma_growth]
        """
        # --- 1. Feature Extraction ---
        # Visual Features: (B, 1280)
        v_ax = self.backbone_ax(img_axial)
        v_cor = self.backbone_cor(img_coronal)

        # Tabular Latent (Prior): (B, 128)
        t_lat = self.tabular_encoder(tabular)

        # --- 2. Alignment ---
        # Project Latent to Visual Dimension: (B, 1280)
        t_align = self.align_norm(self.align_proj(t_lat))

        # --- 3. Fusion ---
        # Sequence: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Transformer (Contextualization)
        seq_out = self.transformer(seq)

        # Unpack
        v_ax_ctx = seq_out[:, 0, :]
        v_cor_ctx = seq_out[:, 1, :]
        t_align_ctx = seq_out[:, 2, :]

        # --- 4. Split Readout ---
        # Visual Stream: Mean of contextualized visual tokens
        h_vis = (v_ax_ctx + v_cor_ctx) / 2.0
        h_vis_small = self.vis_stream_proj(h_vis)  # (B, 64)

        # Tabular Context Stream
        h_ctx_small = self.ctx_stream_proj(t_align_ctx)  # (B, 64)

        # Prior Stream: Raw t_lat (B, 128)

        # Assembly: [Visual(64), Context(64), Prior(128)] -> (B, 256)
        feat_combined = torch.cat([h_vis_small, h_ctx_small, t_lat], dim=1)

        # --- 5. Prediction ---
        out = self.head(feat_combined)

        # Apply activations
        # out[:, 0] is alpha (slope), can be negative -> Linear
        # out[:, 1] is sigma_base -> Softplus
        # out[:, 2] is sigma_growth -> Softplus

        alpha = out[:, 0].unsqueeze(1)
        sigma_base = F.softplus(out[:, 1]).unsqueeze(1)
        sigma_growth = F.softplus(out[:, 2]).unsqueeze(1)

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
