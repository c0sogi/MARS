import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class CI_OP_DS_Net(nn.Module):
    """
    Context-Injected Over-Parameterized Dual-Stream Network (CI-OP-DS Net).

    This architecture utilizes a Parallel Dual-Stream Latent Summation topology:
    1. Stream A: An over-parameterized MLP processing clinical metadata to establish a strong baseline.
    2. Stream B: An EfficientNet-B2 backbone processing CT slices, with 'Context Injection'
       (concatenating time and baseline FVC) before the projection head.

    The outputs are summed in the latent space before the final prediction head.
    """

    def __init__(self):
        super(CI_OP_DS_Net, self).__init__()

        # ====================================================
        # Stream A: Over-Parameterized Clinical Anchor
        # ====================================================
        # Input: Age, Sex, Smoke, RelWeeks, BaseFVC (5 features)
        # Architecture: Linear(5 -> 128) -> ReLU -> Linear(128 -> 64)
        # Purpose: Learns the baseline disease trajectory.
        self.stream_a = nn.Sequential(
            nn.Linear(5, Config.HIDDEN_DIM), nn.ReLU(), nn.Linear(Config.HIDDEN_DIM, 64)
        )

        # ====================================================
        # Stream B: Context-Injected Visual Residual
        # ====================================================
        # Backbone: EfficientNet-B2
        # num_classes=0 ensures we get the Global Average Pooled features
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, in_chans=3
        )

        # EfficientNet-B2 typically has 1408 features after GAP
        self.n_features = self.backbone.num_features

        # Freezing Logic: Unfreeze top two stages; keep bottom frozen
        # 1. Freeze entire backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze Head components (Classifier/ConvHead/BN)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "classifier"):
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True

        # 3. Unfreeze the last 2 blocks of the feature extractor
        # self.backbone.blocks is typically a nn.Sequential
        blocks = list(self.backbone.blocks.children())
        num_blocks = len(blocks)
        for i in range(max(0, num_blocks - 2), num_blocks):
            for param in blocks[i].parameters():
                param.requires_grad = True

        # Context Injection & MLP
        # Input: Visual Features + Context (RelWeeks, BaseFVC)
        # Dimensions: 1408 + 2 = 1410
        # Architecture: Linear(Fused -> 128) -> ReLU -> Linear(128 -> 64)
        # No Dropout is used, as this stream learns a weak residual correction.
        self.stream_b_mlp = nn.Sequential(
            nn.Linear(self.n_features + 2, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, 64),
        )

        # ====================================================
        # Latent Fusion & Shared Head
        # ====================================================
        # Projects the summed latent representation to Mean and Sigma
        self.head = nn.Linear(64, Config.OUT_DIM)

    def forward(self, img, meta_a, meta_b):
        """
        Forward pass of the CI-OP-DS Net.

        Args:
            img (torch.Tensor): CT Image tensor of shape (B, 3, 260, 260).
            meta_a (torch.Tensor): Clinical features for Anchor Stream (B, 5).
                                   [Age, Sex, Smoke, RelWeeks, BaseFVC]
            meta_b (torch.Tensor): Context features for Visual Stream (B, 2).
                                   [RelWeeks, BaseFVC]

        Returns:
            mu (torch.Tensor): Predicted FVC mean (B,).
            sigma (torch.Tensor): Predicted uncertainty (B,).
        """
        # --- Stream A Processing ---
        h_a = self.stream_a(meta_a)  # Shape: (B, 64)

        # --- Stream B Processing ---
        # 1. Feature Extraction
        vis_feat = self.backbone(img)  # Shape: (B, 1408)

        # 2. Context Injection
        # Concatenate visual features with time and baseline FVC
        fused_feat = torch.cat([vis_feat, meta_b], dim=1)  # Shape: (B, 1410)

        # 3. MLP Projection
        h_b = self.stream_b_mlp(fused_feat)  # Shape: (B, 64)

        # --- Latent Fusion ---
        # Summation enforces residual learning
        h_final = h_a + h_b  # Shape: (B, 64)

        # --- Prediction ---
        out = self.head(h_final)  # Shape: (B, 2)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Enforce positivity for sigma using Softplus + epsilon
        # We do not clip at 70 here; clipping is done in metric/post-processing
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma


# Alias for backward compatibility (Cite debug_lesson_11)
DSPRNet = CI_OP_DS_Net
