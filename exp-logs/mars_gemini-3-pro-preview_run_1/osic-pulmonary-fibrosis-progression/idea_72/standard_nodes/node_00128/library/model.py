import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes the 4 raw clinical features (Age, Sex, Smoke, Percent)
    into a Shared Latent Vector (T_lat).
    """

    def __init__(self, input_dim=4, hidden_dim=64, output_dim=Config.LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class AASLNet(nn.Module):
    """
    Attention-Aggregated Shared-Latent Network (AASL-Net).

    Architecture:
    1. Independent Low-Capacity Visual Backbones (EfficientNet-B0) for Axial and Coronal views.
    2. Shared-Latent Tabular Encoder projecting metadata to T_lat.
    3. Contextual Attention fusing Visual and Tabular tokens.
    4. Attention-Aggregated Readout using Tabular token to weigh Visual views.
    5. Balanced Non-Linear Head combining Visual Context and Raw Prior.
    """

    def __init__(self):
        super().__init__()

        # ====================================================
        # 1. Independent Low-Capacity Visual Backbones
        # ====================================================
        # Using EfficientNet-B0 initialized with ImageNet weights.
        # num_classes=0 ensures we get the GAP output (1280-dim) without final classifier.
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        # ====================================================
        # 2. Shared-Latent Tabular Encoder
        # ====================================================
        self.tabular_encoder = TabularEncoder()

        # ====================================================
        # 3. Normalized Bifurcated Flow (Flow A: Alignment)
        # ====================================================
        # Project T_lat (128) -> T_align (1280) to match backbone dim
        self.tabular_projection = nn.Linear(Config.LATENT_DIM, Config.BACKBONE_DIM)
        self.tabular_ln = nn.LayerNorm(Config.BACKBONE_DIM)

        # ====================================================
        # 4. Pre-Norm Symmetric Attention (Contextualization)
        # ====================================================
        # Input Sequence: [V_ax, V_cor, T_align]
        self.attention_ln = nn.LayerNorm(Config.BACKBONE_DIM)
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.BACKBONE_DIM,
            num_heads=8,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # FFN part of Transformer Block
        self.ffn_ln = nn.LayerNorm(Config.BACKBONE_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(Config.BACKBONE_DIM, Config.BACKBONE_DIM * 4),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.BACKBONE_DIM * 4, Config.BACKBONE_DIM),
        )

        # ====================================================
        # 5. Attention-Aggregated Readout & Compression
        # ====================================================
        # Compresses [H_vis, T'_align] (2560) -> H_ctx (128)
        self.compressor = nn.Sequential(
            nn.Linear(Config.BACKBONE_DIM * 2, Config.BACKBONE_DIM // 2),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.BACKBONE_DIM // 2, Config.LATENT_DIM),
            nn.GELU(),
        )

        # ====================================================
        # 6. Balanced Non-Linear Head
        # ====================================================
        # Input: [H_ctx (128), T_lat (128)] -> 256
        # Strictly enforces dimensionality balancing between visual update and clinical prior.
        self.head = nn.Sequential(
            nn.Linear(Config.LATENT_DIM * 2, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

        # ImageNet Normalization constants
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def normalize_images(self, x):
        """Normalizes input tensors (0-1 range) using ImageNet stats."""
        return (x - self.mean) / self.std

    def forward(self, img_ax, img_cor, tabular, meta):
        """
        Args:
            img_ax: (B, 3, 224, 224) Axial Tri-Slab
            img_cor: (B, 3, 224, 224) Coronal Tri-Slab
            tabular: (B, 4) [Age, Sex, Smoke, Percent]
            meta: (B, 2) [Relative_Week, Baseline_FVC]

        Returns:
            pred_fvc: (B,) Predicted FVC for the specific week
            pred_sigma: (B,) Predicted Confidence for the specific week
        """

        # 1. Normalize Images
        img_ax = self.normalize_images(img_ax)
        img_cor = self.normalize_images(img_cor)

        # 2. Feature Extraction
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)

        t_lat = self.tabular_encoder(tabular)  # (B, 128) - "Raw Prior"

        # 3. Alignment
        t_align = self.tabular_projection(t_lat)  # (B, 1280)
        t_align = self.tabular_ln(t_align)

        # 4. Contextualization
        # Stack sequence: [Axial, Coronal, Tabular]
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Pre-Norm Attention
        seq_norm = self.attention_ln(seq)
        attn_out, _ = self.attention(seq_norm, seq_norm, seq_norm)
        seq = seq + attn_out  # Residual

        # Pre-Norm FFN
        seq_norm = self.ffn_ln(seq)
        ffn_out = self.ffn(seq_norm)
        seq = seq + ffn_out  # Residual

        # Unpack Contextualized Tokens
        v_ax_prime = seq[:, 0, :]  # (B, 1280)
        v_cor_prime = seq[:, 1, :]  # (B, 1280)
        t_align_prime = seq[:, 2, :]  # (B, 1280)

        # 5. Attention-Aggregated Readout
        # Use T'_align as Query to weigh V'_ax and V'_cor
        query = t_align_prime.unsqueeze(1)  # (B, 1, 1280)
        keys = torch.stack([v_ax_prime, v_cor_prime], dim=1)  # (B, 2, 1280)

        # Calculate Attention Weights
        scores = torch.bmm(query, keys.transpose(1, 2))  # (B, 1, 2)
        weights = F.softmax(scores, dim=-1)  # (B, 1, 2)

        alpha_ax = weights[:, 0, 0].unsqueeze(1)  # (B, 1)
        alpha_cor = weights[:, 0, 1].unsqueeze(1)  # (B, 1)

        # Weighted Sum of Visual Tokens
        h_vis = alpha_ax * v_ax_prime + alpha_cor * v_cor_prime  # (B, 1280)

        # Joint Compression
        # Concatenate Aggregated Visual + Contextualized Tabular
        combined_ctx = torch.cat([h_vis, t_align_prime], dim=1)  # (B, 2560)
        h_ctx = self.compressor(combined_ctx)  # (B, 128)

        # 6. Final Assembly & Parameter Prediction
        # Dimensionality Balancing: [Update (128), Prior (128)]
        final_vec = torch.cat([h_ctx, t_lat], dim=1)  # (B, 256)

        params = self.head(final_vec)  # (B, 3)

        alpha = params[:, 0]
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 7. Trajectory Projection
        relative_week = meta[:, 0]
        base_fvc = meta[:, 1]

        # FVC = Base + alpha * delta_week
        pred_fvc = base_fvc + alpha * relative_week

        # Sigma = Base + Growth * |delta_week|
        pred_sigma = sigma_base + sigma_growth * torch.abs(relative_week)

        return pred_fvc, pred_sigma
