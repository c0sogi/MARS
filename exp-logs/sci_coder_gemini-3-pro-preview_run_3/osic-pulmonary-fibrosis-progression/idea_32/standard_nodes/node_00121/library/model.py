import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network.
    Cite Lesson 00052: Sums a Linear Residual Stream and a Deep Interaction Stream in Latent Space.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # =====================================================================
        # Stream A: Deep Interaction (Image + Tabular)
        # =====================================================================
        # 1. Image Backbone
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Fine-Tuning Strategy: Unfreeze top two stages
        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            for i in range(max(0, num_blocks - 2), num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        self.img_dim = self.backbone.num_features
        self.tab_dim = 5
        self.latent_dim = 128

        # Deep Projection: Concatenate Image + Tabular -> MLP -> Latent
        # Cite Lesson 00035: Direct concatenation of strong scalar predictors
        self.deep_project = nn.Sequential(
            nn.Linear(self.img_dim + self.tab_dim, 512),
            nn.ReLU(),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(512, self.latent_dim),
        )

        # =====================================================================
        # Stream B: Linear Residual (Baseline FVC + Time)
        # =====================================================================
        # Cite Lesson 00052: Linear stream for autoregressive features
        # Cite Lesson 00060: Over-parameterize linear stream to latent dim
        self.linear_project = nn.Linear(2, self.latent_dim)

        # =====================================================================
        # Shared Head
        # =====================================================================
        # Cite Lesson 00118: Standard initialization (no zero-init)
        self.head = nn.Linear(self.latent_dim, 2)

    def forward(self, img, tab):
        """
        Args:
            img (torch.Tensor): Image tensor (B, 3, 260, 260).
            tab (torch.Tensor): Tabular tensor (B, 5).
                                [BaseFVC, Time, Age, Sex, Smoke]
        """
        # 1. Deep Stream
        img_feat = self.backbone(img)
        deep_in = torch.cat([img_feat, tab], dim=1)
        deep_latent = self.deep_project(deep_in)

        # 2. Linear Stream (Only BaseFVC and Time)
        # Indices 0 and 1 correspond to BaseFVC and RelativeTime
        linear_latent = self.linear_project(tab[:, :2])

        # 3. Latent Space Summation
        fused = deep_latent + linear_latent

        # 4. Final Projection
        out = self.head(fused)

        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-6

        return mu, sigma
