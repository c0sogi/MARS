import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from library.config import Config


class VisualBackbone(nn.Module):
    """
    EfficientNet-B0 backbone initialized with ImageNet weights.
    Extracts features and applies Global Average Pooling to return
    native 1280-dim vectors.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load pretrained EfficientNet-B0
        base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        # Keep only the feature extractor (convolutional layers)
        self.features = base_model.features

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Output dimension for EfficientNet-B0 is 1280
        self.output_dim = 1280

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.features(x)  # (B, 1280, 7, 7)
        x = self.avgpool(x)  # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (B, 1280)
        return x


class TabularEncoder(nn.Module):
    """
    Deep MLP to encode static clinical metadata into a Shared Latent Vector.
    Structure: Linear -> GELU -> Linear -> GELU
    """

    def __init__(self, input_dim, latent_dim):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class NSLHN(nn.Module):
    """
    Normalized Shared-Latent Holistic Network (NSL-HN).
    Combines dual visual backbones with a shared tabular latent space using
    Normalized Bifurcated Flow and Pre-Norm Symmetric Attention.
    """

    def __init__(self):
        super(NSLHN, self).__init__()

        # 1. Independent Low-Capacity Visual Backbones
        self.axial_backbone = VisualBackbone()
        self.coronal_backbone = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        # Input dim is 7 because we exclude 'Weeks' from the 8 features provided by data.py
        # (Weeks, Percent, Age, Sex[2], Smoking[3]) -> Exclude Weeks -> 7
        self.tabular_input_dim = 7
        self.tabular_encoder = TabularEncoder(
            self.tabular_input_dim, Config.TABULAR_LATENT_DIM
        )

        # 3. Normalized Bifurcated Flow (Flow A: Alignment)
        # Project 128 -> 1280 to match visual dim
        self.proj_align = nn.Linear(Config.TABULAR_LATENT_DIM, Config.VISUAL_DIM)
        self.norm_align = nn.LayerNorm(Config.VISUAL_DIM)

        # 4. Pre-Norm Symmetric Attention (Contextualization)
        # Using TransformerEncoderLayer with norm_first=True
        self.fusion_block = nn.TransformerEncoderLayer(
            d_model=Config.VISUAL_DIM,
            nhead=4,
            dim_feedforward=Config.VISUAL_DIM,  # Keep capacity moderate
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 5. Bottleneck Prior-Anchored Head
        # Concatenation of Fused Context (1280) + Raw Latent Prior (128)
        combined_dim = Config.VISUAL_DIM + Config.TABULAR_LATENT_DIM
        bottleneck_dim = 512

        self.head = nn.Sequential(
            nn.Linear(combined_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(bottleneck_dim, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, axial, coronal, tabular, base_fvc, week, base_week):
        """
        Args:
            axial: (B, 3, 224, 224)
            coronal: (B, 3, 224, 224)
            tabular: (B, 8) - [Weeks, Percent, Age, Sex(2), Smoke(3)]
            base_fvc: (B,)
            week: (B,) - Target week
            base_week: (B,) - Baseline week
        """

        # --- 1. Feature Extraction ---
        v_ax = self.axial_backbone(axial)  # (B, 1280)
        v_cor = self.coronal_backbone(coronal)  # (B, 1280)

        # Slice tabular input to exclude 'Weeks' (index 0)
        # We strictly exclude time from the encoder input
        tab_static = tabular[:, 1:]  # (B, 7)
        t_lat = self.tabular_encoder(tab_static)  # (B, 128) Shared Latent Vector

        # --- 2. Normalized Bifurcated Flow ---
        # Flow A: Alignment for Fusion
        t_align = self.proj_align(t_lat)  # (B, 1280)
        t_align = self.norm_align(t_align)  # LayerNorm for stability

        # --- 3. Pre-Norm Symmetric Attention ---
        # Stack tokens: [Axial, Coronal, Tabular_Aligned]
        # Shape: (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Self-Attention
        tokens_out = self.fusion_block(tokens)  # (B, 3, 1280)

        # Holistic Readout: Global Average Pooling over the sequence
        h_fused = torch.mean(tokens_out, dim=1)  # (B, 1280)

        # --- 4. Bottleneck Prior-Anchored Head ---
        # Concatenate Holistic Context with Raw Latent Prior (Flow B)
        # (B, 1280 + 128) -> (B, 1408)
        combined = torch.cat([h_fused, t_lat], dim=1)

        # Predict Parameters
        params = self.head(combined)  # (B, 3)

        alpha = params[:, 0]  # Slope (can be negative)
        sigma_base = F.softplus(params[:, 1])  # Base uncertainty (>0)
        sigma_growth = F.softplus(params[:, 2])  # Uncertainty growth (>0)

        # --- 5. Parametric Inference ---
        # Calculate Delta Time
        dt = week - base_week

        # Linear Trajectory: FVC = Base + alpha * dt
        pred_fvc = base_fvc + alpha * dt

        # Confidence Trajectory: Sigma = Base + Growth * |dt|
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        return pred_fvc, pred_sigma
