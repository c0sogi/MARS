import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CRDSNet(nn.Module):
    """
    Constrained-Residual Dual-Stream Network (CRDS-Net)

    A hybrid CNN-MLP architecture designed for autoregressive forecasting of FVC decline.
    It synthesizes a strong linear autoregressive anchor with a deep residual correction stream,
    enforcing metric constraints directly within the architecture.
    """

    def __init__(self):
        super(CRDSNet, self).__init__()

        # ---------------------------------------------------------------------
        # Constants & Constraints
        # ---------------------------------------------------------------------
        # Calculate the standardized floor for uncertainty to prevent phantom gains.
        # epsilon = 70 / sigma_global
        # This ensures the model cannot predict a sigma lower than the metric's clipping threshold.
        self.epsilon_std = Config.METRIC_CLIP_SIGMA / Config.TARGET_STD

        # ---------------------------------------------------------------------
        # Stream A: Constrained Linear Residual (The Autoregressive Anchor)
        # ---------------------------------------------------------------------
        # Inputs: Baseline_FVC_Scaled (1) + Relative_Time (1) = 2
        # Outputs: mu_linear, sigma_linear (logits)
        # Bias is True to allow the anchor to model global intercepts/offsets.
        # Cite solution_lesson_node_00180: Preserve Affine Capacity in Linear Residual Anchors.
        self.stream_a = nn.Linear(2, 2, bias=True)

        # ---------------------------------------------------------------------
        # Stream B: Deep Interaction Residual (The Correction)
        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B2)
        # We use 3 input channels because we select 3 slices (Anchor + 2 Boundaries)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global Average Pooling
            in_chans=Config.NUM_SLICES,
        )

        # 2. Fine-Tuning Strategy: Unfreeze top 2 stages, freeze rest
        # First, freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last 2 blocks (stages) of the feature extractor
        # In timm EfficientNet, .blocks is a Sequential of stages
        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        # Unfreeze the conv_head and bn2 (Standard EfficientNet top layers)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Projection Layer
        # Projects high-dim backbone output (1408) to bottleneck (64)
        self.img_projector = nn.Linear(
            self.backbone.num_features, Config.PROJECTION_DIM
        )

        # 4. MLP for Residual Learning
        # Input: Projected Image (64) + All Clinical Scalars (6)
        mlp_input_dim = Config.PROJECTION_DIM + Config.TABULAR_INPUT_DIM

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            # No Dropout explicitly to prevent signal dilution of the residual
            nn.Linear(64, 2),  # mu_deep, sigma_deep (logits)
        )

    def forward(self, images, tabular):
        """
        Args:
            images: (Batch, 3, 260, 260) - Preprocessed CT slices
            tabular: (Batch, 6) - Clinical features
                     Cols: [Baseline_FVC_Scaled, Relative_Time, Age_Scaled, Sex, Smoking, Percent]
        Returns:
            preds: (Batch, 2) -> [mu_scaled, sigma_scaled]
        """

        # ---------------------------------------------------------------------
        # Stream A Execution (Autoregressive Anchor)
        # ---------------------------------------------------------------------
        # Isolate Autoregressive inputs: Baseline FVC (idx 0) and Relative Time (idx 1)
        # STRICTLY exclude static metadata (Age, Sex, Smoking, Percent) from this branch
        input_a = tabular[:, [0, 1]]
        out_a = self.stream_a(input_a)
        mu_lin, sigma_lin_logit = out_a[:, 0], out_a[:, 1]

        # ---------------------------------------------------------------------
        # Stream B Execution (Deep Residual)
        # ---------------------------------------------------------------------
        # 1. Image Feature Extraction
        features = self.backbone(images)  # (Batch, 1408)
        img_emb = self.img_projector(features)  # (Batch, 64)

        # 2. Context Injection
        # Concatenate projected image with ALL clinical scalars (including static ones)
        # This allows the model to learn complex interactions (e.g., Smoking + Fibrosis patterns)
        fused = torch.cat([img_emb, tabular], dim=1)  # (Batch, 70)

        # 3. Residual Prediction
        out_b = self.mlp(fused)
        mu_deep, sigma_deep_logit = out_b[:, 0], out_b[:, 1]

        # ---------------------------------------------------------------------
        # Constraint-Aware Fusion
        # ---------------------------------------------------------------------
        # Mean: Additive Residual
        # The final prediction is the Linear Anchor + Deep Correction
        mu_scaled = mu_lin + mu_deep

        # Sigma: Sum logits -> Softplus -> Add Structural Floor
        # We sum the logits from both streams to combine uncertainty estimates.
        # Softplus ensures positivity.
        # Adding epsilon_std enforces the metric's 70ml floor in the standardized space.
        sigma_total_logit = sigma_lin_logit + sigma_deep_logit
        sigma_scaled = F.softplus(sigma_total_logit) + self.epsilon_std

        # Stack for output (Batch, 2)
        return torch.stack([mu_scaled, sigma_scaled], dim=1)
