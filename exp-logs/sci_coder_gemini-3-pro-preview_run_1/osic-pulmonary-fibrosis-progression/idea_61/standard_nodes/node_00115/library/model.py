import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone.
    Uses EfficientNet-B0 to extract features without dimensionality reduction.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0, pretrained on ImageNet
        # num_classes=0 removes the classification head
        # global_pool='avg' applies GAP to return the feature vector directly
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x):
        # Input: (Batch, 3, 224, 224)
        # Output: (Batch, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Projects clinical metadata to a robust shared latent vector.
    """

    def __init__(self, input_dim=7, latent_dim=128):
        super(TabularEncoder, self).__init__()
        # Deep MLP: Linear -> GeLU -> Linear -> GeLU
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        # Input: (Batch, 7)
        # Output: (Batch, 128) -> T_lat
        return self.net(x)


class NBCSLN(nn.Module):
    """
    Non-Linear Balanced-Context Shared-Latent Network.

    Architecture Flow:
    1. Independent Visual Backbones (Axial & Coronal) -> 1280-dim vectors.
    2. Tabular Encoder -> 128-dim Shared Latent Vector (T_lat).
    3. Alignment: Project T_lat to 1280-dim + LayerNorm -> T_align.
    4. Tokenization: Stack [V_ax, V_cor, T_align].
    5. Attention: Pre-Norm Self-Attention Contextualization.
    6. Readout: Global Average Pooling -> H_fused.
    7. Balanced Bottleneck: Compress H_fused to 128-dim -> H_compressed.
    8. Concatenation: [H_compressed, T_lat] -> 256-dim.
    9. Head: Non-Linear MLP -> [alpha, sigma_base, sigma_growth].
    """

    def __init__(self):
        super(NBCSLN, self).__init__()

        # 1. Independent Visual Backbones
        self.backbone_axial = VisualBackbone()
        self.backbone_coronal = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        # Input features: Age, Sex_M, Sex_F, Smoke_Ex, Smoke_Never, Smoke_Curr, Percent (7 total)
        self.tabular_encoder = TabularEncoder(input_dim=7, latent_dim=Config.LATENT_DIM)

        # 3. Fusion Alignment (Flow A)
        # Project Latent (128) -> Backbone Dim (1280)
        self.align_projection = nn.Linear(Config.LATENT_DIM, Config.BACKBONE_DIM)
        # LayerNorm is critical here to match scale of pre-trained visual features
        self.align_norm = nn.LayerNorm(Config.BACKBONE_DIM)

        # 4. Contextualization (Attention)
        # Pre-Norm Transformer Encoder Layer
        # d_model = 1280, nhead = 8 (160 dim per head)
        self.attention_block = nn.TransformerEncoderLayer(
            d_model=Config.BACKBONE_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm stability
        )

        # 5. Balanced-Bottleneck
        # Compress Fused Context (1280) -> Latent Dim (128)
        self.compressor = nn.Linear(Config.BACKBONE_DIM, Config.LATENT_DIM)

        # 6. Non-Linear Head
        # Input: Compressed Context (128) + Shared Latent (128) = 256
        head_input_dim = Config.LATENT_DIM * 2

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, Config.HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(Config.HIDDEN_DIM, 3),  # Output: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_axial, img_coronal, tabular):
        """
        Args:
            img_axial: (B, 3, 224, 224)
            img_coronal: (B, 3, 224, 224)
            tabular: (B, 7)
        Returns:
            Tensor (B, 3): [alpha, sigma_base, sigma_growth]
        """
        # --- 1. Feature Extraction ---
        # Visual: (B, 1280)
        v_ax = self.backbone_axial(img_axial)
        v_cor = self.backbone_coronal(img_coronal)

        # Tabular: (B, 128) -> T_lat (The "Prior")
        t_lat = self.tabular_encoder(tabular)

        # --- 2. Alignment & Tokenization ---
        # Project T_lat to 1280 and Normalize
        t_align = self.align_projection(t_lat)
        t_align = self.align_norm(t_align)

        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # --- 3. Attention Fusion ---
        # Apply Self-Attention
        # (B, 3, 1280)
        attended_tokens = self.attention_block(tokens)

        # --- 4. Holistic Readout ---
        # Global Average Pooling across sequence dimension (dim=1)
        # This fuses the updated visual and tabular contexts
        # (B, 1280)
        h_fused = torch.mean(attended_tokens, dim=1)

        # --- 5. Balanced Bottleneck ---
        # Compress: (B, 1280) -> (B, 128)
        h_compressed = self.compressor(h_fused)

        # Concatenate with original Shared Latent: [h_compressed, t_lat] -> (B, 256)
        # This enforces the "Balanced-Context" design
        combined = torch.cat([h_compressed, t_lat], dim=1)

        # --- 6. Prediction Head ---
        # (B, 3) -> [alpha, sigma_base, sigma_growth]
        raw_out = self.head(combined)

        # Split outputs for specific activation functions
        # alpha: Slope (can be negative, no activation)
        alpha = raw_out[:, 0].unsqueeze(1)

        # sigma_base, sigma_growth: Confidence (must be positive, Softplus)
        sigma_base = F.softplus(raw_out[:, 1].unsqueeze(1))
        sigma_growth = F.softplus(raw_out[:, 2].unsqueeze(1))

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
