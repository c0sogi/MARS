import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SBPDSNet(nn.Module):
    """
    Supervised Bottleneck-Projected Dual-Stream Network (SB-PDS Net).

    Architecture:
    1. Backbone: EfficientNet-B2 (Fine-tuned top stages).
    2. Stream A: Clinical Anchor MLP (Tabular -> Base Prediction).
    3. Stream B: Visual Residual MLP (Image Projection + Tabular -> Residual).
    4. Fusion: Base + Residual.
    """

    def __init__(self):
        super(SBPDSNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B2)
        # ---------------------------------------------------------------------
        # Load pretrained backbone, remove classifier
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,  # Remove default classifier
            global_pool="",  # We handle pooling manually
        )

        # Determine backbone feature dimension (1408 for EfficientNet-B2)
        self.backbone_dim = self.backbone.num_features

        # --- Fine-Tuning Strategy ---
        # Step 1: Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze top layers (Head + Last 2 Blocks)
        # Unfreeze Conv Head and BN2 if they exist (standard in timm efficientnet)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 blocks
        # self.backbone.blocks is a nn.Sequential of blocks
        num_blocks = len(self.backbone.blocks)
        blocks_to_unfreeze = 2
        start_idx = max(0, num_blocks - blocks_to_unfreeze)

        for i in range(start_idx, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Global Pooling and Bottleneck Projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_projector = nn.Linear(self.backbone_dim, Config.BOTTLENECK_DIM)

        # ---------------------------------------------------------------------
        # 2. Stream A: Clinical Anchor (Tabular Only)
        # ---------------------------------------------------------------------
        # Input: [Baseline_FVC, Time, Age, Sex, Smoking] -> 5 features
        input_dim_tabular = len(Config.TABULAR_FEATURES)

        self.stream_a = nn.Sequential(
            nn.Linear(input_dim_tabular, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.BOTTLENECK_DIM),
            nn.Linear(
                Config.BOTTLENECK_DIM, Config.OUTPUT_DIM
            ),  # Outputs: [Mean, Log_Sigma]
        )

        # ---------------------------------------------------------------------
        # 3. Stream B: Visual Residual (Projected Image + Tabular)
        # ---------------------------------------------------------------------
        # Input: Image Projection (64) + Tabular (5)
        input_dim_b = Config.BOTTLENECK_DIM + input_dim_tabular

        self.stream_b_mlp = nn.Sequential(
            nn.Linear(input_dim_b, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_DIM, Config.BOTTLENECK_DIM),
            nn.Dropout(Config.DROPOUT_RATE),
        )
        self.stream_b_out = nn.Linear(Config.BOTTLENECK_DIM, Config.OUTPUT_DIM)

        # --- Zero-Initialization ---
        # Initialize Stream B output to zero so the model starts as the Clinical Anchor
        nn.init.zeros_(self.stream_b_out.weight)
        nn.init.zeros_(self.stream_b_out.bias)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor of shape (B, 3, H, W)
            tabular: Tensor of shape (B, 5)
        Returns:
            dict containing final and base predictions
        """
        # ---------------------------------------------------------------------
        # Image Processing
        # ---------------------------------------------------------------------
        # Extract features: (B, 1408, H/32, W/32)
        features = self.backbone.forward_features(image)

        # Pool: (B, 1408, 1, 1) -> Flatten: (B, 1408)
        pooled = self.global_pool(features).flatten(1)

        # Project to bottleneck: (B, 64)
        img_proj = self.img_projector(pooled)

        # ---------------------------------------------------------------------
        # Stream A: Clinical Anchor
        # ---------------------------------------------------------------------
        # Predict base trajectory from tabular data alone
        out_a = self.stream_a(tabular)
        base_mean = out_a[:, 0]
        base_sigma_logit = out_a[:, 1]

        # ---------------------------------------------------------------------
        # Stream B: Visual Residual
        # ---------------------------------------------------------------------
        # Concatenate projected image features with raw tabular context
        # Shape: (B, 64 + 5)
        combined = torch.cat([img_proj, tabular], dim=1)

        # Predict residuals
        feat_b = self.stream_b_mlp(combined)
        out_b = self.stream_b_out(feat_b)

        resid_mean = out_b[:, 0]
        resid_sigma_logit = out_b[:, 1]

        # ---------------------------------------------------------------------
        # Fusion
        # ---------------------------------------------------------------------
        # Add residual to base
        final_mean = base_mean + resid_mean
        final_sigma_logit = base_sigma_logit + resid_sigma_logit

        # Convert logits to positive sigma values (Confidence)
        # Use Softplus + epsilon for numerical stability
        base_sigma = F.softplus(base_sigma_logit) + 1e-6
        final_sigma = F.softplus(final_sigma_logit) + 1e-6

        return {
            "final_mean": final_mean,
            "final_sigma": final_sigma,
            "base_mean": base_mean,
            "base_sigma": base_sigma,
        }
