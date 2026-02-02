import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class PCDSNet(nn.Module):
    """
    Projected-Context Dual-Stream Network (PCDS-Net)

    A hybrid CNN-MLP architecture that fuses radiological and clinical data using
    a parallel dual-stream topology with latent summation.
    """

    def __init__(self):
        super(PCDSNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch (Fine-Tuned Content-Adaptive 2.5D)
        # ---------------------------------------------------------------------
        # Backbone: EfficientNet-B2
        # num_classes=0 returns the pooled feature vector (Global Average Pooling)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )

        # Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top two convolutional stages (blocks 5 and 6 for B2)
        # and the final head components (conv_head, bn2) to allow domain adaptation
        for name, param in self.backbone.named_parameters():
            if any(
                key in name for key in ["blocks.5.", "blocks.6.", "conv_head.", "bn2."]
            ):
                param.requires_grad = True

        # Bottleneck Projection: 1408 -> 64
        # Projects high-dim visual features to low-dim to prevent noise dominance
        self.bottleneck = nn.Linear(Config.BACKBONE_OUT_DIM, Config.PROJECTION_DIM)

        # ---------------------------------------------------------------------
        # 2. Stream A: Over-Parameterized Clinical Anchor
        # ---------------------------------------------------------------------
        # Input: [BaseFVC, Time, Age, Sex, Smoking] -> Dim 5
        # Note: Config.CLINICAL_INPUT_DIM is 7 in config file (likely for one-hot),
        # but the Data Loader provides 5 features (Ordinal Smoking). We strictly use 5.
        self.clinical_dim = 5

        self.stream_a = nn.Sequential(
            nn.Linear(self.clinical_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # ---------------------------------------------------------------------
        # 3. Stream B: Context-Injected Visual Interaction
        # ---------------------------------------------------------------------
        # Input: Image Projection (64) + Clinical (5) = 69
        # We explicitly provide clinical context to the visual stream
        input_dim_b = Config.PROJECTION_DIM + self.clinical_dim

        self.stream_b = nn.Sequential(
            nn.Linear(input_dim_b, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # ---------------------------------------------------------------------
        # 4. Latent Fusion & Shared Head
        # ---------------------------------------------------------------------
        # Projects fused latent representation to mu and sigma
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, img, clinical):
        """
        Forward pass of PCDS-Net.

        Args:
            img: Tensor of shape (Batch, 3, H, W) - The CT scan slice.
            clinical: Tensor of shape (Batch, 5) - [BaseFVC, Time, Age, Sex, Smoking].

        Returns:
            mu: Predicted FVC (normalized).
            sigma: Predicted Confidence (normalized).
        """
        # --- Image Processing ---
        # Extract deep features from backbone
        img_feat = self.backbone(img)  # (B, 1408)

        # Project to low-dimensional bottleneck
        img_proj = self.bottleneck(img_feat)  # (B, 64)

        # --- Stream A (Clinical Anchor) ---
        # Learns the "Expected Clinical Trajectory" purely from metadata
        stream_a_out = self.stream_a(clinical)  # (B, 64)

        # --- Stream B (Visual Interaction) ---
        # Injects clinical context into the visual stream to learn patient-specific corrections
        # Concatenate image projection and clinical vector
        stream_b_in = torch.cat([img_proj, clinical], dim=1)  # (B, 69)
        stream_b_out = self.stream_b(stream_b_in)  # (B, 64)

        # --- Fusion ---
        # Summation enforces a residual learning paradigm:
        # Final = Clinical_Prior + Visual_Correction
        latent = stream_a_out + stream_b_out  # (B, 64)

        # --- Prediction ---
        out = self.head(latent)  # (B, 2)

        mu = out[:, 0]
        # Use softplus for sigma to ensure positivity
        # Add epsilon for numerical stability
        sigma = F.softplus(out[:, 1]) + 1e-6

        return mu, sigma
