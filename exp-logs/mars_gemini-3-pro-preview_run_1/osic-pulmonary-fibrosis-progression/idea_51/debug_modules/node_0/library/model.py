import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Low-Capacity Visual Backbone using EfficientNet-B0.
    Extracts features from 224x224 RGB images.
    """

    def __init__(
        self, model_name=Config.BACKBONE_NAME, pretrained=Config.BACKBONE_PRETRAINED
    ):
        super().__init__()
        # Load EfficientNet-B0, remove classifier.
        # global_pool='avg' ensures we get a (B, C) vector, effectively Global Average Pooling.
        # num_classes=0 removes the final fully connected layer.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.out_dim = Config.BACKBONE_OUT_DIM

    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)
        # output shape: (Batch, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Projects raw clinical metadata into a shared latent space.
    """

    def __init__(self, input_dim, latent_dim=Config.LATENT_DIM):
        super().__init__()
        # Deep MLP structure: Linear -> GeLU -> Linear -> GeLU
        # Intermediate dimension is set to 2x latent dim for capacity
        hidden_dim = latent_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # x shape: (Batch, 4)
        # output shape: (Batch, 128)
        return self.net(x)


class NSLHN(nn.Module):
    """
    Normalized Shared-Latent Holistic Network (NSL-HN).

    Integrates dual-view visual features with clinical priors using a
    Shared Latent Topology and Normalized Bifurcated Flow.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_axial = VisualBackbone()
        self.backbone_coronal = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        # Input features: Age, Sex, Smoking, Percent
        self.tabular_encoder = TabularEncoder(input_dim=len(Config.TABULAR_COLS))

        # 3. Normalized Bifurcated Flow (Alignment)
        # Projects Shared Latent (128) to Visual Dimension (1280)
        self.tab_projection = nn.Linear(Config.LATENT_DIM, Config.BACKBONE_OUT_DIM)
        # LayerNorm applied immediately after projection for stability
        self.tab_norm = nn.LayerNorm(Config.BACKBONE_OUT_DIM)

        # 4. Pre-Norm Symmetric Attention (Contextualization)
        # We use a Transformer Encoder Layer to handle the fusion
        # norm_first=True implements Pre-Normalization
        self.fusion_block = nn.TransformerEncoderLayer(
            d_model=Config.BACKBONE_OUT_DIM,
            nhead=Config.ATTN_HEADS,
            dim_feedforward=Config.BACKBONE_OUT_DIM * 2,
            dropout=Config.ATTN_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 5. Bottleneck Prior-Anchored Head
        # Concatenates Holistic Fused Vector (1280) + Shared Latent Vector (128)
        fusion_dim = Config.BACKBONE_OUT_DIM + Config.LATENT_DIM

        self.head_bottleneck = nn.Sequential(
            nn.Linear(fusion_dim, Config.HEAD_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Final Projection: Predicts alpha (slope), sigma_base, sigma_growth
        self.head_out = nn.Linear(Config.HEAD_HIDDEN_DIM, 3)

    def forward(self, img_axial, img_coronal, tabular):
        """
        Forward pass of the NSL-HN.

        Args:
            img_axial (torch.Tensor): Axial view batch (B, 3, 224, 224)
            img_coronal (torch.Tensor): Coronal view batch (B, 3, 224, 224)
            tabular (torch.Tensor): Clinical features batch (B, 4)

        Returns:
            torch.Tensor: Predicted parameters (B, 3) -> [alpha, sigma_base, sigma_growth]
        """
        # 1. Visual Extraction
        v_ax = self.backbone_axial(img_axial)  # (B, 1280)
        v_cor = self.backbone_coronal(img_coronal)  # (B, 1280)

        # 2. Tabular Latent Generation
        t_lat = self.tabular_encoder(tabular)  # (B, 128)

        # 3. Alignment Flow
        # Project latent to visual space and normalize
        t_align = self.tab_projection(t_lat)  # (B, 1280)
        t_align = self.tab_norm(t_align)  # (B, 1280)

        # 4. Tokenization & Fusion
        # Stack tokens: [Axial, Coronal, Tabular_Aligned]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Apply Self-Attention
        context = self.fusion_block(tokens)  # (B, 3, 1280)

        # Holistic Readout: Global Average Pooling across all tokens
        h_fused = context.mean(dim=1)  # (B, 1280)

        # 5. Bottleneck Head
        # Concatenate fused context with the original shared latent prior
        # This ensures the clinical prior is preserved explicitly
        combined = torch.cat([h_fused, t_lat], dim=1)  # (B, 1408)

        # Bottleneck processing
        features = self.head_bottleneck(combined)  # (B, 128)

        # Final projection
        raw_out = self.head_out(features)  # (B, 3)

        # 6. Activation Constraints
        # alpha (Slope): Unconstrained
        alpha = raw_out[:, 0].unsqueeze(1)

        # sigma_base: Positive (Softplus)
        sigma_base = F.softplus(raw_out[:, 1].unsqueeze(1))

        # sigma_growth: Positive (Softplus)
        sigma_growth = F.softplus(raw_out[:, 2].unsqueeze(1))

        # Return stacked parameters
        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
