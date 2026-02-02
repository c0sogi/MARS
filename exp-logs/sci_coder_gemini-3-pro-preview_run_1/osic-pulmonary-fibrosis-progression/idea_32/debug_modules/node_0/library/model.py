import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularExpander(nn.Module):
    """
    Projects low-dimensional clinical features into the high-dimensional
    feature space of the visual backbones.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        in_dim = input_dim

        # Intermediate layers with GeLU
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        # Final projection to embedding dimension (Linear)
        layers.append(nn.Linear(in_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SymmetricAttention(nn.Module):
    """
    Applies Self-Attention with Pre-Normalization to allow context fusion
    between visual and tabular tokens.
    """

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Embed_Dim)
        # Pre-Normalization
        x_norm = self.norm(x)

        # Self-Attention (Query, Key, Value)
        # We use the same input for all to implement Self-Attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)

        # Residual connection
        return x + attn_out


class MPVERNet(nn.Module):
    """
    Max-Pooled Visual-Exclusive Residual Network.

    Features:
    1. Dual Independent EfficientNet-B0 Backbones (Axial & Coronal).
    2. Progressive Tabular Expansion.
    3. Symmetric Attention for Contextualization.
    4. Element-wise Max Pooling for Visual Readout.
    5. Parametric Regression Head anchored on Raw Baseline Features.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent High-Fidelity Backbones
        # Branch A: Axial View
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )
        # Branch B: Coronal View
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # Native dimensionality of EfficientNet-B0 is 1280
        self.feature_dim = Config.FEATURE_DIM

        # 2. Progressive Tabular Expansion
        # Input: Age_Norm, Sex_Enc, Smoke_Ex, Smoke_Never, Smoke_Current, Percent_Norm (6 features)
        self.tab_input_dim = 6
        self.tab_expander = TabularExpander(
            input_dim=self.tab_input_dim,
            hidden_dims=Config.TABULAR_HIDDEN_DIMS[:-1],  # e.g. [64, 256]
            output_dim=self.feature_dim,  # 1280
        )

        # 3. Symmetric Attention
        self.attention = SymmetricAttention(embed_dim=self.feature_dim, num_heads=8)

        # 4. Parametric Head
        # Input: V_res (1280) + Raw Tabular (3: Baseline_FVC, Baseline_Percent, Baseline_Age)
        self.head_input_dim = self.feature_dim + 3

        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3),  # Outputs: alpha (slope), sigma_base, sigma_growth
        )

    def forward(
        self, image_axial, image_coronal, tabular_norm, tabular_raw, time_delta
    ):
        """
        Args:
            image_axial: (B, 3, 224, 224)
            image_coronal: (B, 3, 224, 224)
            tabular_norm: (B, 6) - Normalized features for embedding [Age, Sex, Smokes..., Percent]
            tabular_raw: (B, 3) - Raw scalars [Baseline_FVC, Baseline_Percent, Baseline_Age]
            time_delta: (B, 1) - Relative week number (Week - Baseline_Week)

        Returns:
            fvc_pred: (B, 1)
            confidence_pred: (B, 1)
        """

        # --- 1. Feature Extraction ---
        # Visual Features (B, 1280) - Global Average Pooling is implicit in num_classes=0
        v_ax = self.backbone_ax(image_axial)
        v_cor = self.backbone_cor(image_coronal)

        # Tabular Embedding (B, 1280)
        v_tab = self.tab_expander(tabular_norm)

        # --- 2. Contextualization (Symmetric Attention) ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Apply Attention
        tokens_ctx = self.attention(tokens)

        # Unstack updated tokens
        v_ax_ctx = tokens_ctx[:, 0, :]
        v_cor_ctx = tokens_ctx[:, 1, :]
        # v_tab_ctx is discarded for the readout path to prevent bottlenecking

        # --- 3. Max-Pooled Visual-Exclusive Readout ---
        # Element-wise Max Pooling: captures strongest pathology signal from either view
        v_res = torch.max(v_ax_ctx, v_cor_ctx)

        # --- 4. Parametric Prediction ---
        # Concatenate with Raw Tabular Features (Skip Connection)
        # tabular_raw contains: [Baseline_FVC, Baseline_Percent, Baseline_Age]
        head_input = torch.cat([v_res, tabular_raw], dim=1)

        # Predict parameters
        params = self.head(head_input)

        alpha = params[:, 0:1]  # Slope of decline/incline
        sigma_base_raw = params[:, 1:2]
        sigma_growth_raw = params[:, 2:3]

        # Enforce positivity for uncertainty estimates
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        # --- 5. Trajectory Calculation ---
        # Extract Baseline FVC from raw tabular input (index 0)
        baseline_fvc = tabular_raw[:, 0:1]

        # Linear Trajectory: FVC = Baseline + alpha * delta_t
        fvc_pred = baseline_fvc + alpha * time_delta

        # Linear Confidence: Confidence = sigma_base + sigma_growth * |delta_t|
        confidence_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        return fvc_pred, confidence_pred
