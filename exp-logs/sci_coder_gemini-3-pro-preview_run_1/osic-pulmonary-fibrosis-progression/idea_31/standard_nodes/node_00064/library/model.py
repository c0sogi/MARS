import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class EfficientNetBackbone(nn.Module):
    """
    Extracts features using EfficientNet-B0.
    Returns the Global Average Pooled features (1280-dim) without projection.
    """

    def __init__(self, pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        # Load EfficientNet-B0 with appropriate weights
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.net = models.efficientnet_b0(weights=weights)
        except AttributeError:
            # Fallback for older torchvision versions
            self.net = models.efficientnet_b0(pretrained=pretrained)

        # We only need the feature extractor and the pooling layer
        # EfficientNet structure: features -> avgpool -> classifier
        # We discard the classifier.

    def forward(self, x):
        # x: (B, 3, 224, 224)
        x = self.net.features(x)  # (B, 1280, 7, 7)
        x = self.net.avgpool(x)  # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (B, 1280)
        return x


class TabularGLU(nn.Module):
    """
    Projects low-dimensional tabular data to high-dimensional latent space
    using a Gated Linear Unit to control information flow and reduce noise.
    """

    def __init__(self, input_dim, hidden_dim):
        super(TabularGLU, self).__init__()
        # Project to hidden_dim * 2 to split for GLU
        self.fc = nn.Linear(input_dim, hidden_dim * 2)

    def forward(self, x):
        # x: (B, input_dim)
        out = self.fc(x)
        # GLU: split tensor in half, out = a * sigmoid(b)
        return F.glu(out, dim=-1)


class SymmetricAttention(nn.Module):
    """
    Applies Multi-Head Self-Attention with Pre-Normalization.
    Allows visual tokens to be contextualized by the tabular token.
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SymmetricAttention, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, SeqLen, EmbedDim)

        # Pre-Norm
        x_norm = self.norm(x)

        # Self-Attention
        # attn_output: (B, SeqLen, EmbedDim)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)

        # Residual Connection
        return x + self.dropout(attn_output)


class AVRDAN(nn.Module):
    """
    Anchored Visual-Residual Dual-Axis Network.

    Architecture:
    1. Independent Axial/Coronal Backbones (High-Fidelity)
    2. Tabular GLU Projection
    3. Symmetric Attention (Contextualization)
    4. Visual-Exclusive Pooled Readout (Isolation)
    5. Prior-Anchored Parametric Head (Correction)
    """

    def __init__(self):
        super(AVRDAN, self).__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = EfficientNetBackbone(pretrained=Config.BACKBONE_PRETRAINED)
        self.backbone_cor = EfficientNetBackbone(pretrained=Config.BACKBONE_PRETRAINED)

        # 2. Gated Tabular Expansion
        # Input: 7 (Age, Pct, Sex, Smoke), Output: 1280
        self.tab_glu = TabularGLU(Config.TABULAR_INPUT_DIM, Config.FEATURE_DIM)

        # 3. Symmetric Attention
        # Sequence Length: 3 (Axial, Coronal, Tabular)
        self.attention = SymmetricAttention(
            Config.FEATURE_DIM, num_heads=8, dropout=0.1
        )

        # 4. Parametric Head
        # Inputs: Visual Residual (1280) + Raw Priors (3: BaseFVC, BasePct, Age)
        head_input_dim = Config.FEATURE_DIM + 3

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tab_glu_in, tab_skip, delta_week, baseline_fvc):
        """
        Args:
            img_ax: (B, 3, 224, 224) - Axial Tri-Slab
            img_cor: (B, 3, 224, 224) - Coronal Tri-Slab
            tab_glu_in: (B, 7) - Context features for GLU
            tab_skip: (B, 3) - Raw priors for skip connection
            delta_week: (B,) - Weeks relative to baseline
            baseline_fvc: (B,) - Baseline FVC measurement

        Returns:
            fvc_pred: (B,)
            confidence_pred: (B,)
        """
        # Ensure dimensions for broadcasting
        if delta_week.dim() == 1:
            delta_week = delta_week.unsqueeze(1)
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.unsqueeze(1)

        # --- 1. Feature Extraction ---
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)
        v_tab = self.tab_glu(tab_glu_in)  # (B, 1280)

        # --- 2. Contextualization (Symmetric Attention) ---
        # Stack sequence: [Axial, Coronal, Tabular]
        seq = torch.stack([v_ax, v_cor, v_tab], dim=1)  # (B, 3, 1280)

        # Apply Attention
        seq_out = self.attention(seq)  # (B, 3, 1280)

        # --- 3. Visual-Exclusive Pooled Readout ---
        # We strictly isolate the visual tokens (indices 0 and 1)
        # The tabular token (index 2) is used for context but discarded for the residual
        v_ax_prime = seq_out[:, 0, :]
        v_cor_prime = seq_out[:, 1, :]

        # Average Pool the visual views to get a single Visual Residual vector
        v_residual = (v_ax_prime + v_cor_prime) / 2.0  # (B, 1280)

        # --- 4. Prior-Anchored Parametric Head ---
        # Concatenate Visual Residual with Raw Tabular Priors
        # This ensures the strong clinical prior is not diluted
        head_in = torch.cat([v_residual, tab_skip], dim=1)  # (B, 1283)

        # Predict trajectory parameters
        params = self.head(head_in)  # (B, 3)

        alpha = params[:, 0:1]  # Slope
        sigma_base_raw = params[:, 1:2]  # Intercept uncertainty
        sigma_growth_raw = params[:, 2:3]  # Time-dependent uncertainty

        # Enforce positivity for standard deviations
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        # --- 5. Trajectory Calculation ---
        # FVC = Baseline + alpha * delta_week
        fvc_pred = baseline_fvc + alpha * delta_week

        # Confidence = sigma_base + sigma_growth * |delta_week|
        confidence_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        return fvc_pred.squeeze(1), confidence_pred.squeeze(1)
