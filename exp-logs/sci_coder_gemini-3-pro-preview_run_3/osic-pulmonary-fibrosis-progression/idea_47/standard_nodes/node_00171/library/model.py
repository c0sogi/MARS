import torch
import torch.nn as nn
import timm
from library.config import Config


class ClinicalEncoder(nn.Module):
    """
    Stream A: Clinical Latent Encoder.
    Maps clinical features to a latent space.
    Input: 5 Clinical Features
    Output: 64-dim Latent Vector
    """

    def __init__(self, input_dim=5, latent_dim=64):
        super(ClinicalEncoder, self).__init__()
        # Over-parameterized MLP (Cite Lesson 00060)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class VisualEncoder(nn.Module):
    """
    Stream B: Visual Interaction Encoder.
    Maps Images + Tabular Context to a latent space.
    Input: Image (Batch, 3, H, W), Tabular (Batch, 5)
    Output: 64-dim Latent Vector
    """

    def __init__(self, tabular_dim=5, latent_dim=64):
        super(VisualEncoder, self).__init__()

        # Backbone: EfficientNet-B2
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, in_chans=3, num_classes=0
        )
        self.backbone_dim = self.backbone.num_features

        # Unfreezing Logic
        for param in self.backbone.parameters():
            param.requires_grad = False
        if hasattr(self.backbone, "blocks"):
            for stage in self.backbone.blocks[-2:]:
                for param in stage.parameters():
                    param.requires_grad = True
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Bottleneck Projection (Cite Lesson 00146)
        # Project high-dim image features before concatenating with low-dim tabular
        self.projection = nn.Linear(self.backbone_dim, latent_dim)

        # Interaction MLP (Cite Lesson 00139 - Context Visibility)
        # Input: Projected Image (64) + Clinical Scalars (5)
        fused_input_dim = latent_dim + tabular_dim

        # Removed Dropout (Cite Lesson 00126 - Avoid dropout on residual branch)
        self.mlp = nn.Sequential(
            nn.Linear(fused_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, img, tabular):
        # 1. Extract Visual Features
        features = self.backbone(img)  # (Batch, 1408)

        # 2. Project to lower dimension
        proj = self.projection(features)  # (Batch, 64)

        # 3. Context Injection
        fused = torch.cat([proj, tabular], dim=1)  # (Batch, 69)

        # 4. Compute Latent Representation
        out = self.mlp(fused)  # (Batch, 64)
        return out


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (Latent Fusion).
    Fuses Clinical and Visual streams in latent space before final projection.
    Cite Lesson 00052, 00118.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()
        self.latent_dim = 64
        self.clinical_encoder = ClinicalEncoder(input_dim=5, latent_dim=self.latent_dim)
        self.visual_encoder = VisualEncoder(tabular_dim=5, latent_dim=self.latent_dim)

        # Final Head
        # Maps fused latent (64) to Output (2)
        # Standard Initialization (Cite Lesson 00118)
        self.head = nn.Linear(self.latent_dim, 2)

    def forward(self, img, tabular):
        # Stream A: Clinical Latent
        clinical_latent = self.clinical_encoder(tabular)

        # Stream B: Visual Latent
        visual_latent = self.visual_encoder(img, tabular)

        # Latent Fusion (Summation)
        # Acts as a residual connection: Total = Clinical + Correction(Visual)
        fused_latent = clinical_latent + visual_latent

        # Final Projection
        out = self.head(fused_latent)
        return out
