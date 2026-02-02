import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class BPCDSNet(nn.Module):
    """
    Balanced Projected-Context Dual-Stream Network (BPCDS-Net).

    A hybrid CNN-MLP architecture designed for lung function decline prediction.
    It features a parallel dual-stream topology:
    1. Stream A: Clinical Anchor (MLP) - Learns expected trajectory from clinical priors.
    2. Stream B: Visual Interaction (MLP) - Learns residual corrections from CT scans,
       explicitly conditioned on clinical context.

    The image backbone is EfficientNet-B2 with top-stage fine-tuning and a
    bottleneck projection to balance dimensionality with the clinical stream.
    """

    def __init__(self):
        super(BPCDSNet, self).__init__()

        # =====================================================================
        # Image Backbone (Fine-Tuned Content-Adaptive 2.5D)
        # =====================================================================
        # Load pretrained EfficientNet-B2
        # num_classes=0 ensures we get the pooled feature vector (1408 dim)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            in_chans=Config.NUM_SLICES,  # 3 channels (Anchor + 2 Boundaries)
        )

        # Feature dimension for EfficientNet-B2
        self.backbone_dim = self.backbone.num_features  # 1408

        # Freezing Logic: Unfreeze top 2 stages + head, freeze rest
        self._set_backbone_freezing()

        # Bottleneck Projection: 1408 -> 64
        # Projects high-dim visual noise to a compact latent space
        self.img_projector = nn.Linear(self.backbone_dim, Config.LATENT_DIM)

        # =====================================================================
        # Stream A: Over-Parameterized Clinical Anchor
        # =====================================================================
        # Input: 5 Clinical Features [Baseline_FVC, Rel_Time, Age, Sex, Smoking]
        self.clinical_input_dim = 5

        self.stream_a = nn.Sequential(
            nn.Linear(self.clinical_input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # =====================================================================
        # Stream B: Projected-Context Visual Interaction
        # =====================================================================
        # Input: Image Projection (64) + Clinical Context (5) = 69
        self.stream_b_input_dim = Config.LATENT_DIM + self.clinical_input_dim

        self.stream_b = nn.Sequential(
            nn.Linear(self.stream_b_input_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.LATENT_DIM),
        )

        # =====================================================================
        # Shared Head
        # =====================================================================
        # Projects the fused latent state (64) to Mu and Sigma (2)
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def _set_backbone_freezing(self):
        """
        Freezes the entire backbone, then unfreezes the top two convolutional
        stages and the head components to allow domain adaptation.
        """
        # 1. Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze Head components (conv_head, bn2)
        # Note: timm structure usually has conv_head and bn2 as attributes
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Unfreeze Top 2 Stages (last 2 blocks)
        # self.backbone.blocks is a nn.Sequential
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, images, clinical_features):
        """
        Args:
            images: (Batch, 3, 260, 260) - Preprocessed CT slices
            clinical_features: (Batch, 5) - [Baseline_FVC, Rel_Time, Age, Sex, Smoking]

        Returns:
            mu: Predicted FVC (normalized)
            sigma: Predicted Confidence (normalized scale)
        """
        # --- Image Branch ---
        # Extract features: (Batch, 1408)
        img_feats = self.backbone(images)

        # Project to latent dim: (Batch, 64)
        img_latents = self.img_projector(img_feats)

        # --- Stream A: Clinical Anchor ---
        # (Batch, 5) -> (Batch, 64)
        stream_a_out = self.stream_a(clinical_features)

        # --- Stream B: Visual Interaction ---
        # Concatenate Image Latents + Clinical Context
        # (Batch, 64 + 5) -> (Batch, 69)
        combined_features = torch.cat([img_latents, clinical_features], dim=1)

        # (Batch, 69) -> (Batch, 64)
        stream_b_out = self.stream_b(combined_features)

        # --- Fusion ---
        # Residual Summation: Anchor + Visual Correction
        # (Batch, 64)
        h_final = stream_a_out + stream_b_out

        # --- Head ---
        # (Batch, 2)
        output = self.head(h_final)

        # Split into Mu and Sigma
        mu = output[:, 0]
        raw_sigma = output[:, 1]

        # Enforce positivity for Sigma
        # We do not clip to 70 here; clipping is done in post-processing/metric
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma
