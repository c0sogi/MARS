import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes raw clinical features into a Shared Latent Vector (T_lat).
    Architecture: Deep MLP (Linear -> GeLU -> Linear -> GeLU).
    """

    def __init__(self, input_dim, output_dim):
        super(TabularEncoder, self).__init__()
        # Intermediate dimension can be somewhat arbitrary, choosing mid-point or similar
        hidden_dim = 64

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SLADAN(nn.Module):
    """
    Shared-Latent Aligned Dual-Axis Network (SLA-DAN).

    Components:
    1. Two Independent EfficientNet-B0 Backbones (Axial, Coronal).
    2. Shared-Latent Tabular Encoder.
    3. Bifurcated Tabular Flow (Alignment vs Preservation).
    4. Pre-Norm Symmetric Attention Fusion.
    5. Prior-Anchored Prediction Head.
    """

    def __init__(self, cfg=None):
        super(SLADAN, self).__init__()
        if cfg is None:
            cfg = Config()

        self.cfg = cfg

        # 1. Independent Low-Capacity Visual Backbones
        # Using tf_efficientnet_b0_ns, output dim is 1280
        # num_classes=0 removes the classifier, global_pool='' gives features (we handle pooling if needed,
        # but timm usually returns pooled features if global_pool is set.
        # Here we want the vector output of GAP, so global_pool='avg' is appropriate)
        self.backbone_axial = timm.create_model(
            cfg.BACKBONE, pretrained=True, num_classes=0, global_pool="avg"
        )

        self.backbone_coronal = timm.create_model(
            cfg.BACKBONE, pretrained=True, num_classes=0, global_pool="avg"
        )

        visual_dim = cfg.BACKBONE_DIM  # 1280

        # 2. Shared-Latent Tabular Encoder
        self.tabular_encoder = TabularEncoder(
            input_dim=cfg.TABULAR_INPUT_DIM, output_dim=cfg.LATENT_DIM
        )

        # 3. Bifurcated Flow: Alignment Branch
        # Projects T_lat (128) -> T_align (1280)
        self.tabular_align_proj = nn.Linear(cfg.LATENT_DIM, visual_dim)

        # 4. Pre-Norm Symmetric Attention (Contextualization Phase)
        # Input sequence length: 3 (Axial, Coronal, Tabular)
        # Embedding dim: 1280
        self.norm1 = nn.LayerNorm(visual_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=cfg.NUM_HEADS,
            dropout=cfg.ATTN_DROPOUT,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(visual_dim)
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, cfg.FFN_DIM),
            nn.GELU(),
            nn.Dropout(cfg.ATTN_DROPOUT),
            nn.Linear(cfg.FFN_DIM, visual_dim),
            nn.Dropout(cfg.ATTN_DROPOUT),
        )

        # 5. Prior-Anchored Head
        # Concatenates Fused Context (1280) + Shared Latent (128)
        head_input_dim = visual_dim + cfg.LATENT_DIM

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_axial, img_coronal, tabular):
        """
        Args:
            img_axial: (B, 3, 224, 224)
            img_coronal: (B, 3, 224, 224)
            tabular: (B, 9)
        Returns:
            preds: (B, 3) -> [alpha, sigma_base, sigma_growth]
        """
        # 1. Visual Feature Extraction
        # Output: (B, 1280)
        v_ax = self.backbone_axial(img_axial)
        v_cor = self.backbone_coronal(img_coronal)

        # 2. Tabular Encoding -> Shared Latent Vector T_lat
        # Output: (B, 128)
        t_lat = self.tabular_encoder(tabular)

        # 3. Alignment Branch -> T_align
        # Output: (B, 1280)
        t_align = self.tabular_align_proj(t_lat)

        # 4. Attention Fusion
        # Stack tokens: [V_ax, V_cor, T_align] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Pre-Norm Attention Block
        # Residual connection 1
        x_norm = self.norm1(tokens)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        tokens = tokens + attn_out

        # Residual connection 2 (FFN)
        x_norm = self.norm2(tokens)
        ffn_out = self.ffn(x_norm)
        tokens = tokens + ffn_out

        # Global Average Pooling over the sequence
        # (B, 3, 1280) -> (B, 1280)
        h_fused = tokens.mean(dim=1)

        # 5. Prior-Anchored Head
        # Concatenate Holistic Fused Vector + Original Shared Latent
        # (B, 1280) cat (B, 128) -> (B, 1408)
        combined = torch.cat([h_fused, t_lat], dim=1)

        # Predict parameters
        out = self.head(combined)

        # Unpack outputs
        # alpha: unbounded (slope can be negative)
        # sigma_base, sigma_growth: must be positive
        alpha = out[:, 0].view(-1, 1)
        sigma_base = F.softplus(out[:, 1]).view(-1, 1)
        sigma_growth = F.softplus(out[:, 2]).view(-1, 1)

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
