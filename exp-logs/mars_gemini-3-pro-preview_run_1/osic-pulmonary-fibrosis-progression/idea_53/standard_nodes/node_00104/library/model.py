import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Import Config to access hyperparameters
try:
    from library.config import Config
except ImportError:
    import sys

    sys.path.append(".")
    from library.config import Config


class VisualBackbone(nn.Module):
    """
    Low-Capacity Visual Backbone using EfficientNet-B0.
    Extracts 1280-dim features via Global Average Pooling.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # Load model with 3 input channels (RGB)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # We will handle pooling manually or use the feature map directly
        )
        # EfficientNet-B0 output channels is 1280
        self.out_dim = 1280

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        # Extract features: (Batch, 1280, 7, 7)
        features = self.backbone.forward_features(x)

        # Global Average Pooling: (Batch, 1280)
        # Using adaptive avg pool to handle potential resolution changes robustly
        pooled = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        return pooled


class TabularEncoder(nn.Module):
    """
    Encodes raw clinical metadata into a Shared Latent Vector (T_lat).
    Structure: Linear -> GeLU -> Linear -> GeLU
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class NSLHN(nn.Module):
    """
    Normalized Shared-Latent Holistic Network (NSL-HN).

    Architecture:
    1. Independent Visual Backbones (Axial, Coronal)
    2. Shared Latent Tabular Encoder
    3. Normalized Bifurcated Flow (Align + Skip)
    4. Pre-Norm Symmetric Attention
    5. Bottleneck Prior-Anchored Head
    """

    def __init__(self):
        super().__init__()

        # 1. Visual Backbones
        self.backbone_axial = VisualBackbone(Config.BACKBONE_NAME, pretrained=True)
        self.backbone_coronal = VisualBackbone(Config.BACKBONE_NAME, pretrained=True)

        # 2. Shared Latent Tabular Encoder
        # Input features: Percent, Age, Sex, Ex-smoker, Never-smoker, Current-smoker (6 dims)
        self.tabular_input_dim = 6
        self.tabular_encoder = TabularEncoder(
            input_dim=self.tabular_input_dim,
            hidden_dim=Config.HIDDEN_DIM,
            output_dim=Config.LATENT_DIM,
        )

        # 3. Normalized Bifurcated Flow (Flow A: Alignment)
        # Projects 128-dim latent to 1280-dim visual space with LayerNorm
        self.align_projection = nn.Linear(Config.LATENT_DIM, Config.BACKBONE_DIM)
        self.align_norm = nn.LayerNorm(Config.BACKBONE_DIM)

        # 4. Contextualization (Pre-Norm Symmetric Attention)
        # Embedding dim = 1280, Heads = 8 (1280/8 = 160 per head)
        self.attention_block = nn.TransformerEncoderLayer(
            d_model=Config.BACKBONE_DIM,
            nhead=8,
            dim_feedforward=Config.BACKBONE_DIM * 2,  # Standard expansion
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Normalization for stability
        )

        # 5. Bottleneck Prior-Anchored Head
        # Input: Fused Context (1280) + Shared Latent Skip (128) = 1408
        head_input_dim = Config.BACKBONE_DIM + Config.LATENT_DIM

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 3),  # alpha (slope), sigma_base, sigma_growth
        )

    def forward(self, image_axial, image_coronal, tabular, relative_week, baseline_fvc):
        """
        Args:
            image_axial: (B, 3, 224, 224)
            image_coronal: (B, 3, 224, 224)
            tabular: (B, 6)
            relative_week: (B,) - Weeks since baseline
            baseline_fvc: (B,) - Baseline FVC measurement

        Returns:
            pred_fvc: (B,)
            pred_sigma: (B,)
        """
        # --- 1. Feature Extraction ---
        # V_ax: (B, 1280)
        v_ax = self.backbone_axial(image_axial)
        # V_cor: (B, 1280)
        v_cor = self.backbone_coronal(image_coronal)

        # T_lat: (B, 128)
        t_lat = self.tabular_encoder(tabular)

        # --- 2. Normalized Bifurcated Flow ---
        # Flow A: Align for fusion
        t_align = self.align_projection(t_lat)  # (B, 1280)
        t_align = self.align_norm(t_align)  # Normalize to match visual stats

        # --- 3. Contextualization ---
        # Stack tokens: [V_ax, V_cor, T_align] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Self-Attention
        # Output: (B, 3, 1280)
        contextualized_tokens = self.attention_block(tokens)

        # Holistic Readout: Global Average Pooling across tokens
        # (B, 1280)
        h_fused = torch.mean(contextualized_tokens, dim=1)

        # --- 4. Bottleneck Prior-Anchored Head ---
        # Concatenate fused context with raw latent prior (Flow B)
        # (B, 1408)
        combined = torch.cat([h_fused, t_lat], dim=1)

        # Predict parameters
        # out: (B, 3) -> [alpha, sigma_base, sigma_growth]
        params = self.head(combined)

        alpha = params[:, 0]
        sigma_base = params[:, 1]
        sigma_growth = params[:, 2]

        # Apply activations for sigmas (must be positive)
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        # --- 5. Parametric Inference ---
        # FVC = Baseline + alpha * delta_t
        pred_fvc = baseline_fvc + alpha * relative_week

        # Confidence = sigma_base + sigma_growth * |delta_t|
        pred_sigma = sigma_base + sigma_growth * torch.abs(relative_week)

        return pred_fvc, pred_sigma
