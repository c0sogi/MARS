import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import MODEL_NAME, EMBED_DIM, HIDDEN_DIM, DROPOUT


class TabularAlignment(nn.Module):
    """
    Shallow Non-Linear Tabular Alignment.
    Maps scalar metadata (Age, Percent, Sex, Smoking) to the visual semantic space.
    Structure: Linear -> LayerNorm -> GELU -> Linear
    """

    def __init__(self, input_dim=4, output_dim=1280):
        super(TabularAlignment, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PreNormAttentionBlock(nn.Module):
    """
    Pre-Norm Symmetric Attention Block.
    Applies LayerNorm before Self-Attention and FFN for training stability.
    Includes high dropout for capacity regulation.
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.5):
        super(PreNormAttentionBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Pre-Norm Self-Attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual connection

        # Pre-Norm FFN
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # Residual connection
        return x


class CRHDAN(nn.Module):
    """
    Capacity-Regulated Holistic Dual-Axis Network (CR-HDAN).

    Architecture:
    1. Independent EfficientNet-B0 backbones for Axial and Coronal views.
    2. Shallow alignment of tabular features.
    3. Symmetric Self-Attention fusion of [Axial, Coronal, Tabular] tokens.
    4. Holistic Global Average Pooling readout.
    5. Prior-Anchored Parametric Head predicting trajectory parameters.
    """

    def __init__(self):
        super(CRHDAN, self).__init__()

        # 1. Visual Backbones (Low-Capacity Regime)
        # num_classes=0 returns the Global Average Pooled features (1280-dim for B0)
        self.backbone_axial = timm.create_model(
            MODEL_NAME, pretrained=True, num_classes=0
        )
        self.backbone_coronal = timm.create_model(
            MODEL_NAME, pretrained=True, num_classes=0
        )

        # 2. Tabular Alignment
        # Input: Age, Percent, Sex, Smoking (4 features)
        self.tabular_align = TabularAlignment(input_dim=4, output_dim=EMBED_DIM)

        # 3. Contextualization (Fusion)
        self.fusion_block = PreNormAttentionBlock(
            embed_dim=EMBED_DIM, num_heads=4, dropout=DROPOUT
        )

        # 4. Parametric Head
        # Input: Fused Context (1280) + Raw Tabular (4)
        self.head = nn.Sequential(
            nn.Linear(EMBED_DIM + 4, 512),
            nn.GELU(),
            nn.Dropout(0.2),  # Lower dropout in head to preserve signal
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, axial_img, coronal_img, tab_features, weeks_diff, base_fvc):
        """
        Args:
            axial_img: (B, 3, 224, 224)
            coronal_img: (B, 3, 224, 224)
            tab_features: (B, 4) -> [Age, Percent, Sex, Smoking]
            weeks_diff: (B,) -> Time delta from baseline
            base_fvc: (B,) -> Baseline FVC measurement

        Returns:
            pred_fvc: (B,)
            pred_sigma: (B,)
        """
        # 1. Extract Visual Features
        # Output shape: (B, 1280)
        v_ax = self.backbone_axial(axial_img)
        v_cor = self.backbone_coronal(coronal_img)

        # 2. Align Tabular Features
        # Output shape: (B, 1280)
        v_tab = self.tabular_align(tab_features)

        # 3. Tokenization & Fusion
        # Stack tokens: (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Apply Attention
        # Output shape: (B, 3, 1280)
        contextualized_tokens = self.fusion_block(tokens)

        # 4. Holistic Readout (Global Average Pooling)
        # Collapse sequence dimension: (B, 1280)
        fused_vector = torch.mean(contextualized_tokens, dim=1)

        # 5. Prior-Anchored Parametric Prediction
        # Concatenate with raw tabular features (Skip connection for priors)
        # Shape: (B, 1284)
        head_input = torch.cat([fused_vector, tab_features], dim=1)

        # Predict parameters
        # params: (B, 3) -> [alpha, sigma_base_raw, sigma_growth_raw]
        params = self.head(head_input)

        alpha = params[:, 0]
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 6. Trajectory Calculation
        # FVC = Baseline + alpha * delta_t
        pred_fvc = base_fvc + alpha * weeks_diff

        # Sigma = Sigma_base + Sigma_growth * |delta_t|
        pred_sigma = sigma_base + sigma_growth * torch.abs(weeks_diff)

        return pred_fvc, pred_sigma
