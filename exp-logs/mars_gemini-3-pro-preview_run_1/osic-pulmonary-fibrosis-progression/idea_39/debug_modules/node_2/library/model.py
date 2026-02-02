import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ProgressiveTabularMLP(nn.Module):
    """
    Progressive Alignment module to project low-dim tabular features
    into the high-dim visual manifold (1280-dim).
    Structure: Input -> 64 -> 256 -> 1280
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        super().__init__()
        layers = []
        in_d = input_dim

        # Progressive expansion
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_d, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h_dim

        # Final projection to match visual backbone
        layers.append(nn.Linear(in_d, output_dim))
        # No activation on final projection to allow full manifold mapping

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class SymmetricAttention(nn.Module):
    """
    Pre-Norm Self-Attention block to contextualize visual tokens with tabular information.
    Includes Multi-Head Attention and a Feed-Forward Network.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1, ffn_dropout=0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Embed_Dim)

        # Pre-Norm Attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual

        # Pre-Norm FFN
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out  # Residual

        return x


class PAVENet(nn.Module):
    """
    Progressive-Aligned Visual-Exclusive Network (PAVE-Net).

    Components:
    1. Independent EfficientNet-B0 Backbones (Axial & Coronal).
    2. Progressive Tabular MLP.
    3. Symmetric Attention for Contextualization.
    4. Visual-Exclusive Pooling.
    5. Prior-Anchored Parametric Head.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        # Using 'efficientnet_b0' with 1280-dim output
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,  # Remove classifier
            global_pool="avg",  # Global Average Pooling
        )

        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Feature dimension of EfficientNet-B0
        self.embed_dim = Config.BACKBONE_DIM

        # 2. Progressive Tabular Alignment
        self.tabular_mlp = ProgressiveTabularMLP(
            input_dim=Config.TABULAR_INPUT_DIM,
            hidden_dims=Config.TABULAR_HIDDEN_DIMS[:-1],  # [64, 256]
            output_dim=self.embed_dim,  # 1280
            dropout=0.0,
        )

        # 3. Symmetric Attention
        self.attention = SymmetricAttention(
            embed_dim=self.embed_dim,
            num_heads=Config.ATTN_HEADS,
            dropout=Config.ATTN_DROPOUT,
            ffn_dropout=Config.FFN_DROPOUT,
        )

        # 4. Prior-Anchored Parametric Head
        # Input: Visual Residual (1280) + Anchor Features (2: Base_FVC, Base_Percent)
        head_input_dim = self.embed_dim + len(Config.ANCHOR_COLS)

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # Alpha, Sigma_Base, Sigma_Growth
        )

    def forward(self, img_ax, img_cor, tabular, anchor, weeks, raw_base_fvc):
        """
        Args:
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            tabular: (B, 7) - Normalized clinical features
            anchor: (B, 2) - Normalized anchor features (Base_FVC, Base_Percent)
            weeks: (B, 1) - Relative weeks from baseline
            raw_base_fvc: (B, 1) or (B,) - Un-normalized baseline FVC for reconstruction

        Returns:
            fvc_pred: (B, 1)
            sigma_pred: (B, 1)
        """
        batch_size = img_ax.size(0)

        # --- 1. Feature Extraction ---
        # Visual Features (B, 1280)
        feat_ax = self.backbone_ax(img_ax)
        feat_cor = self.backbone_cor(img_cor)

        # Tabular Features (B, 1280)
        feat_tab = self.tabular_mlp(tabular)

        # --- 2. Contextualization ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # Apply Attention
        ctx_tokens = self.attention(tokens)

        # --- 3. Visual-Exclusive Readout ---
        # Extract updated visual tokens (indices 0 and 1)
        ctx_ax = ctx_tokens[:, 0, :]
        ctx_cor = ctx_tokens[:, 1, :]

        # Average Pool Visuals -> Visual Residual (B, 1280)
        visual_residual = (ctx_ax + ctx_cor) / 2.0

        # --- 4. Parametric Prediction ---
        # Concatenate with Anchor Features (Skip Connection)
        # anchor shape is (B, 2)
        head_input = torch.cat([visual_residual, anchor], dim=1)

        # Predict parameters: [Alpha, Sigma_Base, Sigma_Growth]
        params = self.head(head_input)

        alpha = params[:, 0].view(-1, 1)  # Slope
        sigma_b = F.softplus(params[:, 1]).view(-1, 1)  # Base Uncertainty
        sigma_g = F.softplus(params[:, 2]).view(-1, 1)  # Growth Uncertainty

        # --- 5. Trajectory Logic ---
        # FVC = Base + Alpha * Weeks
        # Ensure raw_base_fvc is correct shape
        if isinstance(raw_base_fvc, torch.Tensor):
            raw_base_fvc = raw_base_fvc.view(-1, 1)

        fvc_pred = raw_base_fvc + (alpha * weeks)

        # Confidence = Sigma_Base + Sigma_Growth * |Weeks|
        sigma_pred = sigma_b + (sigma_g * torch.abs(weeks))

        return fvc_pred, sigma_pred
