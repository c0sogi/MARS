import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class TSCRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPR-Net).
    (Class name kept as TSCRNet for compatibility with existing scripts)

    Implements a 'Linear + Deep' residual architecture:
    1. Linear Stream: Captures the strong autoregressive trend (BaseFVC + Time).
       Crucially, this has NO activation to preserve negative Z-scores.
    2. Deep Stream: Captures non-linear corrections from Images and Demographics.
    3. Fusion: Summation in latent space.
    """

    def __init__(self):
        super(TSCRNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch (Fine-Tuned Content-Adaptive 2.5D)
        # ---------------------------------------------------------------------
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            in_chans=Config.IN_CHANNELS,
            global_pool="avg",
        )

        self.img_feature_dim = self.backbone.num_features
        self.img_projector = nn.Linear(self.img_feature_dim, Config.PROJECTION_DIM)
        self._set_backbone_trainable_layers()

        # ---------------------------------------------------------------------
        # 2. Stream A: Linear Residual (The "Wide" Stream)
        # ---------------------------------------------------------------------
        # Input: Baseline FVC, Relative Time (2 features)
        # Cite Lesson 60: Over-parameterize the linear stream to avoid bottlenecks.
        # Cite Lesson 52: Use a linear residual stream for strong autoregressive signals.
        self.linear_stream = nn.Linear(2, Config.HIDDEN_DIM)

        # ---------------------------------------------------------------------
        # 3. Stream B: Deep Interaction (The "Deep" Stream)
        # ---------------------------------------------------------------------
        # Input: Image Projection (64) + All Tabular (5)
        # Purpose: Learn complex non-linear corrections to the linear trend.
        self.deep_input_dim = Config.PROJECTION_DIM + 5

        self.deep_stream = nn.Sequential(
            nn.Linear(self.deep_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, Config.HIDDEN_DIM),
        )

        # ---------------------------------------------------------------------
        # 4. Heads
        # ---------------------------------------------------------------------
        # Mean Head: Projects H_fused (64) -> 1
        self.mean_head = nn.Linear(Config.HIDDEN_DIM, 1)

        # Uncertainty Head: Projects H_fused (64) + |t_rel| (1) -> 1
        # Cite Lesson 55: Direct pathway for uncertainty from simple features.
        self.sigma_head = nn.Linear(Config.HIDDEN_DIM + 1, 1)

    def _set_backbone_trainable_layers(self):
        """
        Freezes the entire backbone, then unfreezes the top two stages.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        trainable_modules = [
            self.backbone.blocks[5],
            self.backbone.blocks[6],
            self.backbone.conv_head,
            self.backbone.bn2,
        ]

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

    def forward(self, image, tabular, time_abs):
        """
        Args:
            image (Tensor): (B, 3, H, W)
            tabular (Tensor): (B, 5) - [BaseFVC, RelTime, Age, Sex, Smoke]
            time_abs (Tensor): (B, 1)
        """
        # --- Image Branch ---
        img_feats = self.backbone(image)
        img_proj = self.img_projector(img_feats)  # (B, 64)

        # --- Stream A: Linear Residual ---
        # Extract [BaseFVC, RelTime] from tabular
        # Note: No ReLU here! Preserves negative Z-scores.
        linear_input = tabular[:, :2]
        h_linear = self.linear_stream(linear_input)  # (B, 64)

        # --- Stream B: Deep Interaction ---
        # Concatenate Image and All Tabular
        deep_input = torch.cat([img_proj, tabular], dim=1)  # (B, 69)
        h_deep = self.deep_stream(deep_input)  # (B, 64)

        # --- Fusion ---
        # Additive Residual: Prediction = LinearTrend + DeepCorrection
        h_fused = h_linear + h_deep  # (B, 64)

        # --- Heads ---
        mu = self.mean_head(h_fused)

        # Uncertainty: Conditioned on fused state (incl. linear trend) and absolute time
        sigma_input = torch.cat([h_fused, time_abs], dim=1)
        raw_sigma = self.sigma_head(sigma_input)
        sigma = F.softplus(raw_sigma) + Config.SIGMA_EPSILON

        return mu, sigma
