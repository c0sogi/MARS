import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TabularEncoder(nn.Module):
    """
    Encodes the 4-dimensional tabular input (Age, Sex, Smoking, Percent)
    into a robust Shared Latent Vector using a Deep MLP.
    """

    def __init__(self, input_dim=4, latent_dim=128):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class BBSLNet(nn.Module):
    """
    Balanced-Bottleneck Shared-Latent Network (BBSL-Net).

    Architecture:
    1. Independent Low-Capacity Visual Backbones (EfficientNet-B0) for Axial and Coronal views.
    2. Shared-Latent Tabular Encoder.
    3. Normalized Bifurcated Flow for alignment.
    4. Pre-Norm Symmetric Attention for fusion.
    5. Balanced-Bottleneck Head with Non-Linear Readout.

    Predicts trajectory parameters (alpha, sigma_base, sigma_growth) and computes
    the final FVC and Confidence based on the time delta.
    """

    def __init__(self):
        super(BBSLNet, self).__init__()

        # Configuration
        self.backbone_name = Config.backbone
        self.backbone_dim = Config.backbone_dim  # 1280 for EfficientNet-B0
        self.latent_dim = Config.latent_dim  # 128

        # ---------------------------------------------------------
        # 1. Independent Visual Backbones
        # ---------------------------------------------------------
        # We use num_classes=0 to get the Global Average Pooled features (Batch, 1280)
        # without the final classifier layer.
        self.backbone_ax = timm.create_model(
            self.backbone_name, pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            self.backbone_name, pretrained=True, num_classes=0
        )

        # ---------------------------------------------------------
        # 2. Shared-Latent Tabular Encoder
        # ---------------------------------------------------------
        self.tabular_encoder = TabularEncoder(input_dim=4, latent_dim=self.latent_dim)

        # ---------------------------------------------------------
        # 3. Normalized Bifurcated Flow (Flow A: Alignment)
        # ---------------------------------------------------------
        # Project 128-dim latent to 1280-dim backbone space
        self.latent_projector = nn.Linear(self.latent_dim, self.backbone_dim)
        # LayerNorm to handle scale disparity between initialized projection and pre-trained feats
        self.latent_norm = nn.LayerNorm(self.backbone_dim)

        # ---------------------------------------------------------
        # 4. Pre-Norm Symmetric Attention (Contextualization)
        # ---------------------------------------------------------
        # Transformer Encoder Layer: Pre-Norm (norm_first=True), Batch First
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.backbone_dim,
            nhead=8,  # 1280 / 8 = 160 dim per head
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention_fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ---------------------------------------------------------
        # 5. Balanced-Bottleneck Head
        # ---------------------------------------------------------
        # Compression: Project 1280-dim fused vector down to 128-dim
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone_dim, self.latent_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Non-Linear Readout
        # Input: Concatenation of Compressed Context (128) + Shared Latent (128) = 256
        # This strictly enforces 50/50 bandwidth between visual context and clinical prior.
        self.head = nn.Sequential(
            nn.Linear(self.latent_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular, time_delta, baseline_fvc):
        """
        Args:
            img_ax: (Batch, 3, 224, 224) Axial Tri-Slab
            img_cor: (Batch, 3, 224, 224) Coronal Tri-Slab
            tabular: (Batch, 4) Normalized tabular features
            time_delta: (Batch,) Time difference (Weeks - Baseline_Week)
            baseline_fvc: (Batch,) FVC at baseline

        Returns:
            preds: (Batch, 2) [Predicted_FVC, Predicted_Confidence]
        """
        # 1. Extract Visual Features
        # Output: (Batch, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # 2. Encode Tabular Features
        # Output: (Batch, 128) -> T_lat
        t_lat = self.tabular_encoder(tabular)

        # 3. Align Tabular Features for Fusion
        # Output: (Batch, 1280)
        t_align = self.latent_projector(t_lat)
        t_align = self.latent_norm(t_align)

        # 4. Attention Fusion
        # Stack tokens: [Axial, Coronal, Tabular_Aligned] -> (Batch, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Apply Transformer
        # Output: (Batch, 3, 1280)
        tokens_out = self.attention_fusion(tokens)

        # Global Average Pooling over the updated tokens -> H_fused
        # Output: (Batch, 1280)
        h_fused = torch.mean(tokens_out, dim=1)

        # 5. Balanced Bottleneck & Prediction
        # Compress fused context -> (Batch, 128)
        h_compressed = self.bottleneck(h_fused)

        # Concatenate with original Shared Latent (Prior Preservation)
        # Output: (Batch, 256)
        combined = torch.cat([h_compressed, t_lat], dim=1)

        # Predict Parameters -> (Batch, 3)
        params = self.head(combined)

        # Unpack parameters
        alpha = params[:, 0]
        # Enforce positivity for confidence parameters
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # 6. Calculate Trajectory (Parametric Inference)
        # FVC = Baseline + alpha * delta
        fvc_pred = baseline_fvc + alpha * time_delta

        # Confidence = sigma_base + sigma_growth * |delta|
        sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

        # Stack predictions for loss calculation: (Batch, 2)
        return torch.stack([fvc_pred, sigma_pred], dim=1)
