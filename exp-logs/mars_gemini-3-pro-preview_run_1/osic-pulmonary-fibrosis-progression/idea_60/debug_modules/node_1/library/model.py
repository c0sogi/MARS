import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes raw tabular features into a Shared Latent Vector (T_lat).
    Architecture: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim, output_dim=128, hidden_dim=256):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class DCSLNet(nn.Module):
    """
    Decoupled-Context Shared-Latent Network (DCSL-Net).

    Key Components:
    1. Independent Visual Backbones (Axial/Coronal EfficientNet-B0).
    2. Shared Latent Tabular Encoder.
    3. Context Fusion via Pre-Norm Self-Attention.
    4. Decoupled Readout (50% Prior, 25% Visual Context, 25% Tabular Context).
    5. Parametric Head (Alpha, Sigma_base, Sigma_growth).
    """

    def __init__(self):
        super(DCSLNet, self).__init__()

        # ==========================================
        # 1. Independent Low-Capacity Visual Backbones
        # ==========================================
        # EfficientNet-B0 outputting 1280-dim vector (num_classes=0 does GAP by default in timm)
        self.backbone_axial = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )
        self.backbone_coronal = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        # ==========================================
        # 2. Shared-Latent Tabular Encoder
        # ==========================================
        # Input dim is 6: [Age_norm, Sex_enc, Smoke_0, Smoke_1, Smoke_2, Percent_norm]
        self.tabular_encoder = TabularEncoder(
            input_dim=6, output_dim=Config.SHARED_LATENT_DIM
        )

        # ==========================================
        # 3. Normalized Bifurcation Flow
        # ==========================================
        # Projects T_lat (128) to T_align (1280) for attention
        self.tab_align_proj = nn.Linear(Config.SHARED_LATENT_DIM, Config.VISUAL_DIM)
        self.tab_align_norm = nn.LayerNorm(Config.VISUAL_DIM)

        # ==========================================
        # 4. Pre-Norm Symmetric Attention (Contextualization)
        # ==========================================
        # Transformer Encoder Layer: d_model=1280
        # norm_first=True enables Pre-Normalization (Stability)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.VISUAL_DIM,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ==========================================
        # 5. Decoupled-Stream Readout
        # ==========================================
        # Visual Context Stream: 1280 -> 64
        self.vis_context_proj = nn.Sequential(
            nn.Linear(Config.VISUAL_DIM, Config.CONTEXT_STREAM_DIM), nn.GELU()
        )

        # Tabular Context Stream: 1280 -> 64
        self.tab_context_proj = nn.Sequential(
            nn.Linear(Config.VISUAL_DIM, Config.CONTEXT_STREAM_DIM), nn.GELU()
        )

        # ==========================================
        # 6. Parametric Head
        # ==========================================
        # Input: [H_vis (64) | H_ctx (64) | T_lat (128)] = 256 dim
        final_dim = (
            Config.CONTEXT_STREAM_DIM
            + Config.CONTEXT_STREAM_DIM
            + Config.SHARED_LATENT_DIM
        )
        self.head = nn.Linear(final_dim, 3)  # alpha, sigma_base, sigma_growth

    def forward(self, img_axial, img_coronal, tabular):
        """
        Args:
            img_axial: (B, 3, 224, 224)
            img_coronal: (B, 3, 224, 224)
            tabular: (B, 6)
        Returns:
            preds: (B, 3) -> [alpha, sigma_base, sigma_growth]
        """
        batch_size = img_axial.shape[0]

        # 1. Visual Extraction (Independent Streams)
        # Output: (B, 1280)
        v_ax = self.backbone_axial(img_axial)
        v_cor = self.backbone_coronal(img_coronal)

        # 2. Tabular Encoding (Shared Latent)
        # Output: T_lat (B, 128)
        t_lat = self.tabular_encoder(tabular)

        # 3. Alignment
        # Project T_lat to T_align (B, 1280) and Normalize
        t_align = self.tab_align_proj(t_lat)
        t_align = self.tab_align_norm(t_align)

        # 4. Context Fusion
        # Stack tokens: [V_ax, V_cor, T_align] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Transformer (Pre-Norm)
        # Output: (B, 3, 1280)
        context_tokens = self.context_fusion(tokens)

        # Unpack tokens
        v_ax_ctx = context_tokens[:, 0, :]
        v_cor_ctx = context_tokens[:, 1, :]
        t_align_ctx = context_tokens[:, 2, :]

        # 5. Decoupled Readout Streams

        # Stream A: Visual Context
        # Mean of contextualized visual tokens
        h_vis_raw = (v_ax_ctx + v_cor_ctx) / 2.0
        h_vis = self.vis_context_proj(h_vis_raw)  # (B, 64)

        # Stream B: Tabular Context
        h_ctx = self.tab_context_proj(t_align_ctx)  # (B, 64)

        # Stream C: Prior Preservation (Raw T_lat)
        # t_lat is (B, 128)

        # Assembly: [H_vis (64), H_ctx (64), T_lat (128)]
        # Total dim: 256.
        # Ratios: 25% Visual Context, 25% Tabular Context, 50% Clinical Prior.
        combined = torch.cat([h_vis, h_ctx, t_lat], dim=1)

        # 6. Parametric Prediction
        out = self.head(combined)

        # Separate outputs
        alpha = out[:, 0].view(-1, 1)  # Slope (can be negative)
        sigma_base = out[:, 1].view(-1, 1)  # Base uncertainty
        sigma_growth = out[:, 2].view(-1, 1)  # Uncertainty growth rate

        # Enforce positivity for sigmas using Softplus
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
