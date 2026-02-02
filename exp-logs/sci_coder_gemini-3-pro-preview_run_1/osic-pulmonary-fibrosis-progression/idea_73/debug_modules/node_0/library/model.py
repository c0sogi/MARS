import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone (EfficientNet-B0).
    Extracts features using Global Average Pooling without projection to maintain high fidelity.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0 with ImageNet weights
        # global_pool='avg' ensures we get a vector [B, 1280]
        # num_classes=0 removes the classification head
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x):
        # Input: [B, 3, 224, 224]
        # Output: [B, 1280]
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Projects clinical scalars to a robust latent vector (T_lat).
    Structure: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self):
        super(TabularEncoder, self).__init__()
        # Input: 7 -> Hidden (64) -> Output (128)
        self.net = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, 64),
            nn.GELU(),
            nn.Linear(64, Config.LATENT_DIM),
            nn.GELU(),
        )

    def forward(self, x):
        # Input: [B, 7]
        # Output: [B, 128]
        return self.net(x)


class PCCGNet(nn.Module):
    """
    Prior-Centric Context-Gated Network (PCCG-Net).
    Features:
    - Independent Dual-View Backbones
    - Shared Latent Tabular Topology
    - Normalized Bifurcated Flow
    - Pre-Norm Symmetric Attention
    - Prior-Gated Aggregation
    - Balanced Non-Linear Readout
    """

    def __init__(self):
        super(PCCGNet, self).__init__()

        # 1. Visual Backbones (Axial & Coronal)
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        # 2. Tabular Encoder
        self.tabular_encoder = TabularEncoder()

        # 3. Bifurcated Flow A: Alignment
        # Project T_lat (128) -> T_align (1280) + LayerNorm
        self.project_align = nn.Linear(Config.LATENT_DIM, Config.ALIGN_DIM)
        self.ln_align = nn.LayerNorm(Config.ALIGN_DIM)

        # 4. Pre-Norm Symmetric Attention (Contextualization)
        # Sequence: [V_ax, V_cor, T_align]
        # Transformer Encoder Layer with Pre-Norm (norm_first=True) for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.ALIGN_DIM,
            nhead=Config.NUM_HEADS,
            dim_feedforward=Config.FFN_DIM,
            dropout=Config.ATTN_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 5. Prior-Gated Aggregation Bottleneck
        # Input: Concat(Weighted_Vis, T'_align) -> 1280 + 1280 = 2560
        # Output: Context Vector (H_ctx) -> 128
        self.bottleneck = nn.Sequential(
            nn.Linear(Config.ALIGN_DIM * 2, Config.CONTEXT_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # 6. Balanced Non-Linear Head
        # Input: Concat(H_ctx, T_lat) -> 128 + 128 = 256
        # Strictly enforces balanced contribution from Learned Context and Raw Prior
        self.head = nn.Sequential(
            nn.Linear(Config.CONTEXT_DIM + Config.LATENT_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, x_axial, x_coronal, x_tabular, delta_week, base_fvc):
        """
        Forward pass implementing the PCCG-Net logic and Anchored Trajectory prediction.

        Args:
            x_axial: [B, 3, 224, 224] - Axial Tri-Slab
            x_coronal: [B, 3, 224, 224] - Coronal Tri-Slab
            x_tabular: [B, 7] - Clinical features
            delta_week: [B] - Relative week number
            base_fvc: [B] - Baseline FVC measurement

        Returns:
            preds: [B, 2] - Predicted (FVC, Confidence)
        """
        # --- 1. Feature Extraction ---
        v_ax = self.backbone_ax(x_axial)  # [B, 1280]
        v_cor = self.backbone_cor(x_coronal)  # [B, 1280]
        t_lat = self.tabular_encoder(x_tabular)  # [B, 128] (Raw Prior)

        # --- 2. Bifurcated Flow A: Alignment ---
        t_align = self.project_align(t_lat)  # [B, 1280]
        t_align = self.ln_align(t_align)  # [B, 1280]

        # --- 3. Contextualization (Fusion) ---
        # Stack tokens: [V_ax, V_cor, T_align]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # [B, 3, 1280]

        # Apply Transformer
        ctx_tokens = self.context_transformer(tokens)  # [B, 3, 1280]

        v_ax_prime = ctx_tokens[:, 0, :]  # [B, 1280]
        v_cor_prime = ctx_tokens[:, 1, :]  # [B, 1280]
        t_align_prime = ctx_tokens[:, 2, :]  # [B, 1280] (Contextualized Tabular)

        # --- 4. Prior-Gated Aggregation ---
        # Query: T'_align, Keys: [V'_ax, V'_cor]
        query = t_align_prime.unsqueeze(1)  # [B, 1, 1280]
        keys = torch.stack([v_ax_prime, v_cor_prime], dim=1)  # [B, 2, 1280]

        # Calculate attention weights
        scores = torch.matmul(query, keys.transpose(1, 2))  # [B, 1, 2]
        attn_weights = F.softmax(scores, dim=-1)  # [B, 1, 2]

        # Compute Weighted Visual Context (H_vis)
        h_vis = torch.matmul(attn_weights, keys).squeeze(1)  # [B, 1280]

        # Concatenate H_vis with Contextualized Tabular Token
        h_full = torch.cat([h_vis, t_align_prime], dim=1)  # [B, 2560]

        # Project to Context Vector (H_ctx)
        h_ctx = self.bottleneck(h_full)  # [B, 128]

        # --- 5. Balanced Readout ---
        # Assembly: Concat Context (128) + Raw Prior (128) -> [B, 256]
        z = torch.cat([h_ctx, t_lat], dim=1)

        # Predict Parameters
        out = self.head(z)  # [B, 3]

        alpha = out[:, 0]
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        # --- 6. Anchored Trajectory Logic ---
        # FVC = Baseline + alpha * (Week - Baseline_Week)
        # Confidence = Sigma_base + Sigma_growth * |Week - Baseline_Week|

        fvc_pred = base_fvc + alpha * delta_week
        confidence = sigma_base + sigma_growth * torch.abs(delta_week)

        # Return stacked predictions [B, 2]
        return torch.stack([fvc_pred, confidence], dim=1)
