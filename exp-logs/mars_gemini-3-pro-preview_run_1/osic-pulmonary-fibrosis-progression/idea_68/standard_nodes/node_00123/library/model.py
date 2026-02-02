import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes tabular features into a shared latent vector.
    Architecture: Linear -> GELU -> Linear -> GELU
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PGARNet(nn.Module):
    """
    Prior-Guided Attention-Readout Network (PGAR-Net).

    Architecture Design:
    1. Independent Low-Capacity Visual Backbones: Two EfficientNet-B0s for Axial and Coronal views.
       Features are extracted via Global Average Pooling (GAP) without compression (1280-dim).
    2. Shared-Latent Tabular Encoder: Deep MLP projecting clinical data to a robust latent vector (128-dim).
    3. Normalized Bifurcated Flow: Projects tabular latent to visual dimension (1280-dim) with LayerNorm.
    4. Pre-Norm Symmetric Attention: Contextualizes [Axial, Coronal, Tabular] tokens using a Transformer layer.
    5. Prior-Guided Readout: Uses the contextualized tabular token to dynamically weigh visual tokens.
    6. Non-Linear Parametric Head: Predicts trajectory parameters (alpha, sigma_base, sigma_growth).
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Visual Backbones
        # ==========================================
        # Two independent backbones for Axial and Coronal views.
        # global_pool='avg' and num_classes=0 ensures we get the 1280-dim feature vector directly.
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        self.vis_dim = Config.VISUAL_FEATURE_DIM  # 1280

        # ==========================================
        # 2. Tabular Encoder
        # ==========================================
        # Input: Age(1) + Percent(1) + Sex(2) + Smoking(3) = 7
        self.tab_input_dim = 7
        self.tab_latent_dim = Config.TABULAR_LATENT_DIM  # 128
        self.tab_encoder = TabularEncoder(self.tab_input_dim, self.tab_latent_dim)

        # ==========================================
        # 3. Bifurcated Flow & Alignment
        # ==========================================
        # Flow A: Project tabular latent to visual dimension for attention alignment
        self.flow_a_proj = nn.Linear(self.tab_latent_dim, self.vis_dim)
        # LayerNorm to prevent initialization shock in attention block
        self.flow_a_norm = nn.LayerNorm(self.vis_dim)

        # ==========================================
        # 4. Attention Mechanism (Contextualization)
        # ==========================================
        # Transformer Encoder Layer with Pre-Norm (norm_first=True)
        # Sequence: [Axial, Coronal, Tabular_Aligned]
        self.context_block = nn.TransformerEncoderLayer(
            d_model=self.vis_dim,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # ==========================================
        # 5. Readout & Assembly
        # ==========================================
        self.context_hidden_dim = Config.CONTEXT_HIDDEN_DIM  # 64

        # Projections for the interaction streams (Compression)
        self.vis_stream_proj = nn.Linear(self.vis_dim, self.context_hidden_dim)
        self.tab_stream_proj = nn.Linear(self.vis_dim, self.context_hidden_dim)

        # ==========================================
        # 6. Prediction Head
        # ==========================================
        # Input: Vis_Stream(64) + Tab_Stream(64) + Tab_Raw(128) = 256
        # This balance ensures the raw prior (50% of vector) remains dominant.
        self.head_input_dim = (2 * self.context_hidden_dim) + self.tab_latent_dim

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, axial, coronal, tabular, dt=None, base_fvc=None):
        """
        Args:
            axial: (B, 3, 224, 224) Axial Tri-Slab images
            coronal: (B, 3, 224, 224) Coronal Tri-Slab images
            tabular: (B, 7) Normalized tabular features
            dt: (B,) Relative week (optional, for trajectory calculation)
            base_fvc: (B,) Baseline FVC (optional, for trajectory calculation)

        Returns:
            If dt and base_fvc are provided:
                fvc_pred, sigma_pred
            Else:
                alpha, sigma_base, sigma_growth
        """
        # --- 1. Feature Extraction ---
        v_ax = self.backbone_ax(axial)  # (B, 1280)
        v_cor = self.backbone_cor(coronal)  # (B, 1280)
        t_lat = self.tab_encoder(tabular)  # (B, 128)

        # --- 2. Alignment ---
        # Project tabular to visual space and norm (Flow A)
        t_align = self.flow_a_norm(self.flow_a_proj(t_lat))  # (B, 1280)

        # --- 3. Contextualization ---
        # Stack sequence: [Axial, Coronal, Tabular]
        # Shape: (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Pass through Transformer Layer (Pre-Norm Self-Attention)
        seq_out = self.context_block(seq)

        # Unpack tokens
        v_ax_prime = seq_out[:, 0, :]  # (B, 1280)
        v_cor_prime = seq_out[:, 1, :]  # (B, 1280)
        t_align_prime = seq_out[:, 2, :]  # (B, 1280)

        # --- 4. Prior-Guided Readout ---
        # Use Tabular token to attend to Visual tokens
        # Query: t_align_prime (B, 1, 1280)
        query = t_align_prime.unsqueeze(1)

        # Keys: [v_ax_prime, v_cor_prime] (B, 2, 1280)
        keys = torch.stack([v_ax_prime, v_cor_prime], dim=1)

        # Attention Scores: Q * K^T -> (B, 1, 2)
        attn_scores = torch.bmm(query, keys.transpose(1, 2))

        # Attention Weights (Softmax)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Weighted Visual Context: Weights * Keys -> (B, 1, 1280)
        h_vis = torch.bmm(attn_weights, keys).squeeze(1)

        # --- 5. Compression & Assembly ---
        vis_stream = self.vis_stream_proj(h_vis)  # (B, 64)
        tab_stream = self.tab_stream_proj(t_align_prime)  # (B, 64)

        # Concatenate: [Vis(64), Tab_Context(64), Tab_Raw(128)]
        # Raw t_lat is preserved (Flow B)
        combined = torch.cat([vis_stream, tab_stream, t_lat], dim=1)  # (B, 256)

        # --- 6. Prediction ---
        out = self.head(combined)

        # Extract parameters
        alpha = out[:, 0]  # Slope (can be negative)
        sigma_base = F.softplus(out[:, 1])  # Base confidence (must be positive)
        sigma_growth = F.softplus(out[:, 2])  # Growth confidence (must be positive)

        # --- 7. Trajectory Calculation (if inputs provided) ---
        if dt is not None and base_fvc is not None:
            # FVC = Baseline + alpha * dt
            fvc_pred = base_fvc + alpha * dt

            # Sigma = Base + Growth * |dt|
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            return fvc_pred, sigma_pred

        return alpha, sigma_base, sigma_growth
