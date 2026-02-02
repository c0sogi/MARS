import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Model as ModelConfig


class PyramidFeatureExtractor(nn.Module):
    """
    Extracts multi-scale features from an EfficientNet backbone.
    Focuses on Strides 8, 16, and 32 to capture texture and structure.
    """

    def __init__(self):
        super().__init__()
        # We enforce indices [2, 3, 4] for EfficientNet-B0 to match
        # the requirement of Strides 8, 16, and 32.
        # Index 2 -> Stride 8
        # Index 3 -> Stride 16
        # Index 4 -> Stride 32
        target_indices = [2, 3, 4]

        self.backbone = timm.create_model(
            ModelConfig.BACKBONE,
            pretrained=ModelConfig.PRETRAINED,
            features_only=True,
            out_indices=target_indices,
        )

        # Dynamically determine output channel dimensions
        dummy_input = torch.zeros(1, 3, 224, 224)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        self.out_channels = [f.shape[1] for f in features]

    def forward(self, x):
        """
        Args:
            x: Input image tensor (B, 3, H, W)
        Returns:
            list: List of pooled feature vectors (B, C_i)
        """
        # Extract features maps
        features = self.backbone(x)

        # Apply Global Average Pooling to each level
        # (B, C, H, W) -> (B, C)
        pooled_features = [f.mean(dim=(2, 3)) for f in features]

        return pooled_features


class PyramidDualAxisNet(nn.Module):
    """
    Pyramid Dual-Axis Attention Network.
    Fuses multi-view (Axial/Coronal) and multi-scale visual features
    with clinical metadata using a Transformer.
    """

    def __init__(self):
        super().__init__()

        self.embed_dim = ModelConfig.EMBED_DIM

        # --- 1. Visual Backbones ---
        self.axial_extractor = PyramidFeatureExtractor()
        self.coronal_extractor = PyramidFeatureExtractor()

        # --- 2. Feature Projections ---
        # Project variable channel dimensions to fixed EMBED_DIM

        # Axial Projections (one per pyramid level)
        self.axial_projs = nn.ModuleList(
            [nn.Linear(c, self.embed_dim) for c in self.axial_extractor.out_channels]
        )

        # Coronal Projections (one per pyramid level)
        self.coronal_projs = nn.ModuleList(
            [nn.Linear(c, self.embed_dim) for c in self.coronal_extractor.out_channels]
        )

        # Tabular Projection
        # Input: 4 features (Age, Sex, SmokingStatus, Percent)
        self.tabular_proj = nn.Sequential(
            nn.Linear(4, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        # --- 3. Symmetric Pyramid Attention ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=ModelConfig.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=ModelConfig.DROPOUT,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=ModelConfig.NUM_LAYERS
        )

        # --- 4. Parametric Head ---
        # Input: Transformed Tabular Token + Raw Tabular Embedding (Skip Connection)
        # Output: alpha (slope), sigma_base, sigma_growth
        self.head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, ModelConfig.OUTPUT_DIM),
        )

    def forward(self, axial, coronal, tabular, time_delta, baseline_fvc):
        """
        Args:
            axial: (B, 3, 224, 224) Axial Tri-Slab
            coronal: (B, 3, 224, 224) Coronal Tri-Slab
            tabular: (B, 4) Normalized tabular features
            time_delta: (B, 1) Weeks from baseline
            baseline_fvc: (B, 1) Baseline FVC measurement

        Returns:
            fvc_pred: (B, 1) Predicted FVC
            sigma_pred: (B, 1) Predicted Confidence
        """
        batch_size = axial.size(0)

        # 1. Extract Visual Features
        ax_feats = self.axial_extractor(axial)  # List of 3 vectors
        co_feats = self.coronal_extractor(coronal)  # List of 3 vectors

        # 2. Project and Tokenize
        tokens = []

        # Axial Tokens
        for i, feat in enumerate(ax_feats):
            tokens.append(self.axial_projs[i](feat))

        # Coronal Tokens
        for i, feat in enumerate(co_feats):
            tokens.append(self.coronal_projs[i](feat))

        # Tabular Token
        tab_embed = self.tabular_proj(tabular)  # (B, D)
        tokens.append(tab_embed)

        # Stack tokens: (B, 7, D)
        # Sequence: [Ax_L1, Ax_L2, Ax_L3, Co_L1, Co_L2, Co_L3, Tabular]
        x = torch.stack(tokens, dim=1)

        # 3. Attention Mechanism
        x = self.transformer(x)

        # Extract the refined tabular token (last in sequence)
        tab_out = x[:, -1, :]  # (B, D)

        # 4. Prediction Head with Skip Connection
        # Concatenate refined token with original clinical embedding
        head_input = torch.cat([tab_out, tab_embed], dim=1)  # (B, 2*D)

        params = self.head(head_input)  # (B, 3)

        # Unpack parameters
        alpha = params[:, 0:1]  # Slope
        sigma_base = F.softplus(params[:, 1:2])  # Base uncertainty (positive)
        sigma_growth = F.softplus(params[:, 2:3])  # Uncertainty growth (positive)

        # 5. Parametric Inference
        # FVC = Baseline + alpha * delta
        fvc_pred = baseline_fvc + alpha * time_delta

        # Confidence = sigma_base + sigma_growth * |delta|
        sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        return fvc_pred, sigma_pred
