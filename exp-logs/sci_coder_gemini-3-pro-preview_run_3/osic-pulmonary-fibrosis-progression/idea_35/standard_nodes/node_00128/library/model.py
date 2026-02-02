import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class UDSRNet(nn.Module):
    """
    Unregularized Dual-Stream Residual Network (UDSR-Net).

    Structure:
    1. Image Branch: EfficientNet-B2 (Top layers unfrozen) -> Global Features
    2. Stream A: Clinical Anchor MLP (Input: Tabular) -> Base prediction
    3. Stream B: Visual Residual MLP (Input: Image + Tabular) -> Residual correction

    Fusion:
    - Mu = Mu_base + Mu_residual
    - Sigma = Softplus(Sigma_base_raw + Sigma_residual_raw)
    """

    def __init__(self):
        super(UDSRNet, self).__init__()

        # =====================================================================
        # 1. Image Branch (Backbone)
        # =====================================================================
        # Load EfficientNet-B2
        # num_classes=0 and global_pool='' ensures we get spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=Config.SLICES_PER_PATIENT,
        )

        # Determine feature dimension dynamically
        with torch.no_grad():
            dummy = torch.randn(
                1, Config.SLICES_PER_PATIENT, Config.IMG_SIZE, Config.IMG_SIZE
            )
            features = self.backbone(dummy)
            self.img_feat_dim = features.shape[1]  # Typically 1408 for B2

        # --- Freeze/Unfreeze Logic ---
        # 1. Freeze entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze top layers (Head + Last 2 Blocks)
        # Unfreeze Head components
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 blocks of the 'blocks' container
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Iterate through the last 2 blocks
            for i in range(max(0, num_blocks - 2), num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Global Average Pooling to collapse spatial dims
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # =====================================================================
        # 2. Stream A: Clinical Anchor (The Foundation)
        # =====================================================================
        # Input: [Baseline_FVC, Relative_Time, Age, Sex, Smoking] -> 5 features
        self.tabular_dim = 5

        self.stream_a = nn.Sequential(
            nn.Linear(self.tabular_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 2),  # Outputs: mu_base, raw_sigma_base
        )

        # =====================================================================
        # 3. Stream B: Visual Residual Stream (The Correction)
        # =====================================================================
        # Input: Image Features + Tabular Features (Early Fusion)
        self.stream_b_input_dim = self.img_feat_dim + self.tabular_dim

        # Explicitly NO Dropout to preserve weak visual signals
        self.stream_b = nn.Sequential(
            nn.Linear(self.stream_b_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 2),  # Outputs: delta_mu, delta_raw_sigma
        )

    def forward(self, image, tabular):
        """
        Forward pass of UDSR-Net.

        Args:
            image (torch.Tensor): Batch of images (B, 3, H, W)
            tabular (torch.Tensor): Batch of tabular data (B, 5)
                                    [Baseline_FVC, Relative_Time, Age, Sex, Smoking]

        Returns:
            mu_final (torch.Tensor): Predicted FVC mean (B,)
            sigma_final (torch.Tensor): Predicted FVC confidence (B,)
        """
        # --- Image Branch ---
        # Extract features
        x_img = self.backbone(image)  # (B, C, H', W')
        x_img = self.pooling(x_img)  # (B, C, 1, 1)
        x_img = x_img.flatten(1)  # (B, C)

        # --- Stream A (Clinical Anchor) ---
        # Predicts the baseline trajectory based solely on clinical data
        out_a = self.stream_a(tabular)
        mu_base = out_a[:, 0]
        raw_sigma_base = out_a[:, 1]

        # --- Stream B (Visual Residual) ---
        # Predicts corrections based on image + clinical data
        x_combined = torch.cat([x_img, tabular], dim=1)
        out_b = self.stream_b(x_combined)
        delta_mu = out_b[:, 0]
        delta_raw_sigma = out_b[:, 1]

        # --- Probabilistic Output Fusion ---
        # Mean: Additive residual correction
        mu_final = mu_base + delta_mu

        # Uncertainty: Additive in logit space, then Softplus
        # This allows the visual stream to increase or decrease uncertainty
        # relative to the clinical baseline.
        sigma_final = F.softplus(raw_sigma_base + delta_raw_sigma) + 1e-6

        return mu_final, sigma_final
