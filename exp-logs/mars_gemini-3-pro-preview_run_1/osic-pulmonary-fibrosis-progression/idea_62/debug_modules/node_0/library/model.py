import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes raw tabular features into a Shared Latent Vector (T_lat).
    Structure: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class NDSSLN(nn.Module):
    """
    Non-Linear Decoupled-Stream Shared-Latent Network (NDS-SLN).

    Architecture:
    1. Two Independent EfficientNet-B0 Backbones (Axial & Coronal) -> 1280-dim each.
    2. Shared-Latent Tabular Encoder -> 128-dim (T_lat).
    3. Normalized Bifurcated Flow:
       - Flow A: T_lat -> Projected to 1280 + LayerNorm -> T_align.
       - Flow B: T_lat preserved as Prior.
    4. Pre-Norm Symmetric Attention (Fusion):
       - Input: [V_ax, V_cor, T_align]
       - Output: Contextualized tokens.
    5. Decoupled-Stream Compressed Readout:
       - Visual Stream: Mean(V'_ax, V'_cor) -> Bottleneck -> 64-dim.
       - Tabular Stream: T'_align -> Bottleneck -> 64-dim.
       - Prior Stream: Raw T_lat -> 128-dim.
       - Assembly: Concat -> 256-dim.
    6. Non-Linear Parametric Head -> (alpha, sigma_base, sigma_growth).
    """

    def __init__(self):
        super(NDSSLN, self).__init__()

        # ==========================
        # 1. Visual Backbones
        # ==========================
        # Independent backbones for Axial and Coronal views
        # num_classes=0 ensures we get the pooled feature vector (1280 for B0)
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        # Feature dimension for EfficientNet-B0
        self.vis_dim = Config.VISUAL_DIM  # 1280

        # ==========================
        # 2. Tabular Encoder
        # ==========================
        # Input features: Age_norm, Sex_bin, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent_norm (6 features)
        self.tabular_input_dim = 6
        self.latent_dim = Config.LATENT_DIM  # 128
        self.tab_encoder = TabularEncoder(
            input_dim=self.tabular_input_dim,
            hidden_dim=64,
            output_dim=self.latent_dim,
        )

        # ==========================
        # 3. Bifurcation & Alignment
        # ==========================
        # Project T_lat (128) to Visual Dim (1280) for attention
        self.align_proj = nn.Linear(self.latent_dim, self.vis_dim)
        self.align_norm = nn.LayerNorm(self.vis_dim)

        # ==========================
        # 4. Contextualization
        # ==========================
        # Single Transformer Encoder Layer with Pre-Normalization
        # d_model = 1280, nhead = 8 (divisible), dim_feedforward = 2048 (default)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.vis_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm
        )
        self.context_block = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ==========================
        # 5. Decoupled Readout
        # ==========================
        self.bottleneck_dim = Config.BOTTLENECK_DIM  # 64

        # Visual Context Stream Bottleneck
        self.vis_bottleneck = nn.Linear(self.vis_dim, self.bottleneck_dim)

        # Tabular Context Stream Bottleneck
        self.ctx_bottleneck = nn.Linear(self.vis_dim, self.bottleneck_dim)

        # ==========================
        # 6. Parametric Head
        # ==========================
        # Input: Visual(64) + Context(64) + Prior(128) = 256
        self.head_input_dim = (
            self.bottleneck_dim + self.bottleneck_dim + self.latent_dim
        )

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular):
        """
        Args:
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            tabular: (B, 6)
        Returns:
            alpha: (B, 1) Slope of decline
            sigma_base: (B, 1) Base uncertainty
            sigma_growth: (B, 1) Uncertainty growth rate
        """
        batch_size = img_ax.size(0)

        # 1. Visual Extraction
        # Output: (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # 2. Tabular Encoding (Prior)
        # Output: (B, 128)
        t_lat = self.tab_encoder(tabular)

        # 3. Bifurcation Flow A: Alignment
        # Output: (B, 1280)
        t_align = self.align_proj(t_lat)
        t_align = self.align_norm(t_align)

        # 4. Contextualization (Fusion)
        # Stack tokens: [V_ax, V_cor, T_align] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Attention
        # Output: (B, 3, 1280)
        contextualized = self.context_block(tokens)

        # Unstack
        v_ax_prime = contextualized[:, 0, :]
        v_cor_prime = contextualized[:, 1, :]
        t_align_prime = contextualized[:, 2, :]

        # 5. Decoupled-Stream Compression
        # Visual Stream: Mean of visual tokens
        h_vis_raw = (v_ax_prime + v_cor_prime) / 2.0
        h_vis = self.vis_bottleneck(h_vis_raw)  # (B, 64)

        # Tabular Context Stream
        h_ctx = self.ctx_bottleneck(t_align_prime)  # (B, 64)

        # Prior Stream: Raw T_lat (B, 128)

        # Assembly
        # [Visual(64), Context(64), Prior(128)] -> (B, 256)
        assembled = torch.cat([h_vis, h_ctx, t_lat], dim=1)

        # 6. Parametric Prediction
        out = self.head(assembled)  # (B, 3)

        alpha = out[:, 0].view(-1, 1)
        sigma_base = out[:, 1].view(-1, 1)
        sigma_growth = out[:, 2].view(-1, 1)

        # Enforce positivity constraints on sigma
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        return alpha, sigma_base, sigma_growth
