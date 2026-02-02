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
        # Outputs: Latent Representation (64)
        # Cite Lesson 00060: Over-parameterize the linear stream by projecting to latent space.
        self.stream_a = nn.Linear(2, Config.PROJECTION_DIM, bias=False)

        # ---------------------------------------------------------------------
        # Stream B: Deep Interaction Residual (The Correction)
        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B2)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,  # Remove classification head
            global_pool="avg",  # Global Average Pooling
            in_chans=Config.NUM_SLICES,
        )

        # 2. Fine-Tuning Strategy: Unfreeze top 2 stages, freeze rest
        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Projection Layer
        self.img_projector = nn.Linear(
            self.backbone.num_features, Config.PROJECTION_DIM
        )

        # 4. MLP for Residual Learning
        # Input: Projected Image (64) + All Clinical Scalars (6)
        mlp_input_dim = Config.PROJECTION_DIM + Config.TABULAR_INPUT_DIM

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, Config.PROJECTION_DIM),  # Output matches latent dim (64)
        )

        # ---------------------------------------------------------------------
        # Shared Head
        # ---------------------------------------------------------------------
        # Cite Lesson 00055: Do not isolate strong priors from uncertainty.
        # Fused Latent (64) -> Output (2)
        self.head = nn.Linear(Config.PROJECTION_DIM, 2)

    def forward(self, images, tabular):
        """
        Args:
            images: (Batch, 3, 260, 260) - Preprocessed CT slices
            tabular: (Batch, 6) - Clinical features
        Returns:
            preds: (Batch, 2) -> [mu_scaled, sigma_scaled]
        """

        # ---------------------------------------------------------------------
        # Stream A Execution (Autoregressive Anchor)
        # ---------------------------------------------------------------------
        # Isolate Autoregressive inputs: Baseline FVC (idx 0) and Relative Time (idx 1)
        input_a = tabular[:, [0, 1]]
        latent_a = self.stream_a(input_a)  # (Batch, 64)

        # ---------------------------------------------------------------------
        # Stream B Execution (Deep Residual)
        # ---------------------------------------------------------------------
        features = self.backbone(images)  # (Batch, 1408)
        img_emb = self.img_projector(features)  # (Batch, 64)

        # Context Injection
        fused_input = torch.cat([img_emb, tabular], dim=1)  # (Batch, 70)
        latent_b = self.mlp(fused_input)  # (Batch, 64)

        # ---------------------------------------------------------------------
        # Latent Fusion & Prediction
        # ---------------------------------------------------------------------
        # Cite Lesson 00052: Sum in latent space before final projection
        latent_sum = latent_a + latent_b

        logits = self.head(latent_sum)
        mu_scaled, sigma_logit = logits[:, 0], logits[:, 1]

        # Sigma: Softplus + Structural Floor
        sigma_scaled = F.softplus(sigma_logit) + self.epsilon_std

        # Stack for output (Batch, 2)
        return torch.stack([mu_scaled, sigma_scaled], dim=1)
