import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Extracts high-fidelity visual features using EfficientNet-B0.
    Initialized with ImageNet weights.
    Outputs a 1280-dim vector (Global Average Pooling applied natively).
    """

    def __init__(self):
        super().__init__()
        # Create EfficientNet-B0.
        # num_classes=0 removes the classifier and returns the pooled feature vector.
        # in_chans=3 corresponds to the RGB Tri-Slab input.
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, in_chans=3
        )

    def forward(self, x):
        # x: (Batch, 3, 224, 224)
        # Output: (Batch, 1280)
        return self.backbone(x)


class TabularExpander(nn.Module):
    """
    Multi-Stage Dense Tabular Expansion.
    Projects 6-dim clinical scalars to 1280-dim dense vector.
    Structure: Input(6) -> Linear(64) -> GeLU -> Linear(256) -> GeLU -> Linear(1280).
    """

    def __init__(self):
        super().__init__()
        input_dim = Config.TABULAR_INPUT_DIM
        hidden_dims = Config.TABULAR_HIDDEN_DIMS  # [64, 256, 1280]

        layers = []
        # Stage 1: 6 -> 64
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.GELU())

        # Stage 2: 64 -> 256
        layers.append(nn.Linear(hidden_dims[0], hidden_dims[1]))
        layers.append(nn.GELU())

        # Stage 3: 256 -> 1280
        # We do not apply activation after the final projection to allow
        # the vector to populate the full semantic space before attention.
        layers.append(nn.Linear(hidden_dims[1], hidden_dims[2]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: (Batch, 6)
        # Output: (Batch, 1280)
        return self.mlp(x)


class SymmetricAttention(nn.Module):
    """
    Symmetric Self-Attention Fusion.
    Processes the sequence [Axial, Coronal, Tabular] via Multi-Head Attention.
    Aggregates outputs via Global Average Pooling.
    """

    def __init__(self, embed_dim=1280, num_heads=8, dropout=0.1):
        super().__init__()
        # batch_first=True ensures input shape is (Batch, Seq, Feature)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (Batch, SeqLen=3, EmbedDim=1280)

        # Self-Attention
        # attn_output: (Batch, SeqLen, EmbedDim)
        attn_output, _ = self.attn(x, x, x)

        # Residual connection + Normalization
        x = self.norm(x + attn_output)

        # Global Average Pooling over the sequence dimension (dim=1)
        # Aggregates info from all views/modalities into a single context vector.
        # Output: (Batch, 1280)
        pooled = x.mean(dim=1)
        return pooled


class PriorAnchoredHead(nn.Module):
    """
    Regression head with Prior-Preserving Skip Connection.
    Concatenates the fused deep vector with raw tabular priors.
    Predicts: alpha (slope), sigma_base (intercept uncertainty), sigma_growth (slope uncertainty).
    """

    def __init__(self, embed_dim=1280, prior_dim=6):
        super().__init__()
        # Input dimension = Deep Features (1280) + Raw Priors (6)
        self.input_dim = embed_dim + prior_dim

        # Single linear layer allows the model to learn a direct linear combination
        # of the raw priors (e.g., FVC ~ w1*Age + w2*Percent) adjusted by deep features.
        self.head = nn.Linear(self.input_dim, 3)

    def forward(self, fusion_vec, raw_priors):
        # fusion_vec: (Batch, 1280)
        # raw_priors: (Batch, 6)

        # Concatenate via skip connection
        combined = torch.cat([fusion_vec, raw_priors], dim=1)

        # Predict raw logits
        out = self.head(combined)

        # Extract components
        alpha = out[:, 0]  # Slope of decline (can be negative)
        sigma_base = out[:, 1]  # Uncertainty at baseline
        sigma_growth = out[:, 2]  # Uncertainty growth rate

        # Enforce positivity constraints on sigma
        sigma_base = F.softplus(sigma_base)
        sigma_growth = F.softplus(sigma_growth)

        return alpha, sigma_base, sigma_growth


class DPSDAN(nn.Module):
    """
    Dense-Projection Symmetric Dual-Axis Network (DP-SDAN).
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        # Branch A: Axial
        self.backbone_axial = VisualBackbone()
        # Branch B: Coronal
        self.backbone_coronal = VisualBackbone()

        # 2. Multi-Stage Dense Tabular Expansion
        self.tabular_expander = TabularExpander()

        # 3. Symmetric Self-Attention Fusion
        self.fusion = SymmetricAttention(embed_dim=Config.EMBED_DIM, num_heads=8)

        # 4. Prior-Anchored Head
        self.head = PriorAnchoredHead(
            embed_dim=Config.EMBED_DIM, prior_dim=Config.TABULAR_INPUT_DIM
        )

    def forward(self, img_axial, img_coronal, tab_dense):
        """
        Args:
            img_axial: (Batch, 3, 224, 224) - Axial Tri-Slab
            img_coronal: (Batch, 3, 224, 224) - Coronal Tri-Slab
            tab_dense: (Batch, 6) - Normalized clinical features

        Returns:
            alpha: (Batch,) - Predicted slope
            sigma_base: (Batch,) - Predicted baseline confidence
            sigma_growth: (Batch,) - Predicted confidence growth
        """
        # 1. Extract Visual Features
        # (Batch, 1280)
        v_ax = self.backbone_axial(img_axial)
        v_cor = self.backbone_coronal(img_coronal)

        # 2. Expand Tabular Features
        # (Batch, 1280)
        v_tab = self.tabular_expander(tab_dense)

        # 3. Stack for Attention
        # Sequence: [Axial, Coronal, Tabular]
        # Shape: (Batch, 3, 1280)
        seq = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # 4. Fuse
        # Shape: (Batch, 1280)
        fused = self.fusion(seq)

        # 5. Predict
        # Pass both the fused vector and the raw tabular inputs for the skip connection
        alpha, sigma_base, sigma_growth = self.head(fused, tab_dense)

        return alpha, sigma_base, sigma_growth
