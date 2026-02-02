import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes the 4-dim tabular prior (Age, Percent, Sex, Smoking) into a latent vector.
    Architecture: Linear -> GELU -> Linear -> GELU
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PGANet(nn.Module):
    """
    Prior-Gated Aggregation Network (PGA-Net).

    Features:
    - Dual Independent EfficientNet-B0 Backbones (Axial/Coronal)
    - Shared Latent Tabular Encoder
    - Pre-Norm Symmetric Attention for Contextualization
    - Prior-Gated Aggregation for Visual Feature Selection
    - Parametric Trajectory Head (Alpha, Sigma_Base, Sigma_Growth)
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Low-Capacity Visual Backbones
        # We use EfficientNet-B0 initialized with ImageNet weights.
        # num_classes=0 with global_pool='avg' returns the 1280-dim feature vector.
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # 2. Shared-Latent Tabular Encoder
        # Input: 4 features (Age, Percent, Sex, Smoking)
        # Output: T_lat (128 dim)
        self.tabular_encoder = TabularEncoder(
            input_dim=4, output_dim=Config.SHARED_LATENT_DIM
        )

        # 3. Normalized Bifurcated Flow
        # Projects T_lat (128) to T_align (1280) to match visual backbones
        self.to_align = nn.Linear(Config.SHARED_LATENT_DIM, Config.ALIGN_DIM)
        self.align_norm = nn.LayerNorm(Config.ALIGN_DIM)

        # 4. Pre-Norm Symmetric Attention
        # Fuses [V_ax, V_cor, T_align]
        # We use batch_first=True for convenience
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.ALIGN_DIM,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm for stability
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 5. Prior-Gated Readout & Compression
        # Projects 1280-dim streams down to 64-dim
        self.compress_vis = nn.Linear(Config.ALIGN_DIM, Config.COMPRESS_DIM)
        self.compress_int = nn.Linear(Config.ALIGN_DIM, Config.COMPRESS_DIM)

        # 6. Non-Linear Parametric Head
        # Input: T_lat (128) + Vis_Stream (64) + Int_Stream (64) = 256
        self.head = nn.Sequential(
            nn.Linear(Config.ASSEMBLED_DIM, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular, meta):
        """
        Args:
            img_ax (torch.Tensor): Axial view images (B, 3, 224, 224)
            img_cor (torch.Tensor): Coronal view images (B, 3, 224, 224)
            tabular (torch.Tensor): Tabular features (B, 4)
            meta (torch.Tensor): Metadata for inference (B, 2) -> [Baseline_FVC, Delta_Week]

        Returns:
            pred_fvc (torch.Tensor): Predicted FVC (B,)
            pred_sigma (torch.Tensor): Predicted Confidence (B,)
        """
        # --- Visual Features ---
        # Shape: (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # --- Tabular Latent ---
        # Shape: (B, 128)
        t_lat = self.tabular_encoder(tabular)

        # --- Alignment Flow ---
        # Project and Norm: (B, 1280)
        t_align = self.align_norm(self.to_align(t_lat))

        # --- Contextualization (Attention) ---
        # Stack tokens: [V_ax, V_cor, T_align] -> Shape: (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Self-Attention
        ctx_seq = self.attention(seq)

        # Unpack Contextualized Tokens
        v_ax_ctx = ctx_seq[:, 0, :]
        v_cor_ctx = ctx_seq[:, 1, :]
        t_align_ctx = ctx_seq[:, 2, :]

        # --- Prior-Gated Aggregation ---
        # We use the contextualized tabular token to weigh the visual tokens
        # Stack visual context: (B, 2, 1280)
        v_stack_ctx = torch.stack([v_ax_ctx, v_cor_ctx], dim=1)

        # Compute Attention Weights: Softmax(T'_align dot [V'_ax, V'_cor]^T)
        # (B, 1, 1280) @ (B, 1280, 2) -> (B, 1, 2)
        scores = torch.bmm(t_align_ctx.unsqueeze(1), v_stack_ctx.transpose(1, 2))
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted Sum: (B, 1, 2) @ (B, 2, 1280) -> (B, 1, 1280)
        h_vis = torch.bmm(attn_weights, v_stack_ctx).squeeze(1)

        # --- Stream Compression ---
        vis_stream = self.compress_vis(h_vis)  # (B, 64)
        int_stream = self.compress_int(t_align_ctx)  # (B, 64)

        # --- Assembly ---
        # Concatenate: [Raw Prior (128), Visual Context (64), Interaction Context (64)]
        # Result Shape: (B, 256)
        feat = torch.cat([t_lat, vis_stream, int_stream], dim=1)

        # --- Parametric Prediction ---
        out = self.head(feat)

        # Extract parameters
        alpha = out[:, 0]  # Slope (can be negative)
        sigma_base = F.softplus(out[:, 1])  # Base confidence (must be positive)
        sigma_growth = F.softplus(out[:, 2])  # Growth confidence (must be positive)

        # --- Inference Logic ---
        # FVC = Baseline + alpha * delta_t
        # Sigma = Sigma_base + Sigma_growth * |delta_t|
        baseline_fvc = meta[:, 0]
        delta_week = meta[:, 1]

        pred_fvc = baseline_fvc + alpha * delta_week
        pred_sigma = sigma_base + sigma_growth * torch.abs(delta_week)

        return pred_fvc, pred_sigma
