import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone.
    Uses EfficientNet-B0 initialized with ImageNet weights.
    Extracts high-fidelity features (1280-dim) without projection.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0
        # num_classes=0 removes the classifier, global_pool='avg' ensures GAP output
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x):
        # Input: (Batch, 3, 224, 224)
        # Output: (Batch, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Deep MLP projecting scalars to a robust Shared Latent Vector (T_lat).
    """

    def __init__(self, input_dim, latent_dim):
        super(TabularEncoder, self).__init__()
        hidden_dim = latent_dim * 2
        # Deep MLP: Linear -> GeLU -> Linear -> GeLU
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # Input: (Batch, 6)
        # Output: (Batch, 128)
        return self.net(x)


class PCCGNet(nn.Module):
    """
    Prior-Centric Context-Gated Network (PCCG-Net).

    Key Features:
    1. Independent Visual Backbones for Axial and Coronal views.
    2. Bifurcated Flow for Tabular Prior (Alignment vs Preservation).
    3. Pre-Norm Symmetric Attention for Contextualization.
    4. Prior-Gated Aggregation Readout.
    5. Balanced Non-Linear Head for Parametric Inference.
    """

    def __init__(self):
        super(PCCGNet, self).__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        visual_dim = Config.BACKBONE_DIM  # 1280

        # 2. Shared-Latent Tabular Encoder
        # Input features: Age, Sex, Percent, Smoking (3 dims) -> Total 6
        tabular_input_dim = 6
        latent_dim = Config.TABULAR_LATENT_DIM  # 128
        self.tabular_encoder = TabularEncoder(tabular_input_dim, latent_dim)

        # 3. Bifurcated Flow - Alignment Path
        # Project T_lat (128) to T_align (1280) matching visual dim
        self.align_proj = nn.Linear(latent_dim, visual_dim)
        self.align_norm = nn.LayerNorm(visual_dim)

        # 4. Pre-Norm Symmetric Attention (Contextualization Phase)
        self.attn_norm1 = nn.LayerNorm(visual_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=visual_dim,
            num_heads=Config.ATTENTION_HEADS,
            dropout=Config.DROPOUT_RATE,
            batch_first=True,
        )

        self.attn_norm2 = nn.LayerNorm(visual_dim)
        self.ffn = nn.Sequential(
            nn.Linear(visual_dim, Config.FFN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.FFN_DIM, visual_dim),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # 5. Prior-Gated Aggregation Readout
        # Bottleneck MLP to project concatenated features (2560) down to Context (128)
        # Input: Weighted Visual (1280) + Contextualized Tabular (1280)
        self.bottleneck = nn.Sequential(
            nn.Linear(visual_dim * 2, Config.CONTEXT_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # 6. Balanced Non-Linear Head
        # Input: Context Vector (128) + Raw Prior T_lat (128) = 256
        head_input_dim = Config.CONTEXT_DIM + latent_dim
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular, weeks, base_fvc):
        """
        Args:
            img_ax (Tensor): Axial images (B, 3, 224, 224)
            img_cor (Tensor): Coronal images (B, 3, 224, 224)
            tabular (Tensor): Tabular features (B, 6)
            weeks (Tensor): Relative weeks from baseline (B,)
            base_fvc (Tensor): Baseline FVC values (B,)

        Returns:
            Tensor: Predictions (B, 2) -> [FVC, Confidence]
        """

        # --- 1. Feature Extraction ---
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)

        # Get Shared Latent Vector (Raw Prior)
        t_lat = self.tabular_encoder(tabular)  # (B, 128)

        # --- 2. Bifurcated Flow: Alignment ---
        t_align = self.align_proj(t_lat)  # (B, 1280)
        t_align = self.align_norm(t_align)  # Prevent initialization shock

        # --- 3. Contextualization (Attention) ---
        # Stack tokens: [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Pre-Norm Self-Attention
        tokens_norm = self.attn_norm1(tokens)
        attn_out, _ = self.attention(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out

        # Feed-Forward Network
        tokens_norm = self.attn_norm2(tokens)
        ffn_out = self.ffn(tokens_norm)
        tokens = tokens + ffn_out

        # Unpack Contextualized Tokens
        v_ax_prime = tokens[:, 0, :]  # (B, 1280)
        v_cor_prime = tokens[:, 1, :]  # (B, 1280)
        t_align_prime = tokens[:, 2, :]  # (B, 1280)

        # --- 4. Prior-Gated Aggregation ---
        # Use Contextualized Tabular (Query) to weight Visual Tokens (Keys)
        # Query: (B, 1, 1280)
        query = t_align_prime.unsqueeze(1)
        # Keys: (B, 2, 1280)
        keys = torch.stack([v_ax_prime, v_cor_prime], dim=1)

        # Attention Scores: Q * K^T -> (B, 1, 2)
        scores = torch.bmm(query, keys.transpose(1, 2))
        weights = F.softmax(scores, dim=-1)  # [alpha_ax, alpha_cor]

        # Weighted Visual Context: weights * Keys -> (B, 1, 1280)
        h_vis = torch.bmm(weights, keys).squeeze(1)

        # Concatenate Weighted Visual + Contextualized Tabular
        h_full = torch.cat([h_vis, t_align_prime], dim=1)  # (B, 2560)

        # Bottleneck Projection
        h_ctx = self.bottleneck(h_full)  # (B, 128)

        # --- 5. Balanced Head ---
        # Concatenate Context Vector + Raw Prior (Preservation Path)
        # Enforces 50/50 contribution balance
        final_vec = torch.cat([h_ctx, t_lat], dim=1)  # (B, 256)

        # Predict Parameters
        params = self.head(final_vec)  # (B, 3)

        alpha = params[:, 0]
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # --- 6. Parametric Inference ---
        # Calculate final predictions based on anchored trajectory logic

        # FVC = Baseline + alpha * relative_weeks
        fvc_pred = base_fvc + alpha * weeks

        # Confidence = sigma_base + sigma_growth * |relative_weeks|
        confidence_pred = sigma_base + sigma_growth * torch.abs(weeks)

        # Stack results: (B, 2)
        return torch.stack([fvc_pred, confidence_pred], dim=1)
