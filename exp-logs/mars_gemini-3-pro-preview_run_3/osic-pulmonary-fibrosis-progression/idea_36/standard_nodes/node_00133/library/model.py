import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite Lesson 00052: Dual-Stream Residuals for Strong Autoregressive Signals.
    Fuses a Linear Autoregressive Stream with a Deep Interaction Stream in Latent Space.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # --- Stream A: Linear Residual (Base FVC + Time) ---
        # Projects strong autoregressive features to latent space
        # Cite Lesson 00060: Over-Parameterization of Linear Baselines
        self.linear_stream = nn.Linear(2, Config.LATENT_DIM)

        # --- Stream B: Deep Interaction (Image + Full Tabular) ---
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=3,
        )

        # Fine-Tuning Strategy (Cite Lesson 00027)
        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        self.vis_dim = self.backbone.num_features

        # Deep stream components
        self.vis_projector = nn.Linear(self.vis_dim, 256)
        self.tab_projector = nn.Linear(Config.TABULAR_INPUT_DIM, 128)

        # Deep Fusion MLP
        self.deep_fusion = nn.Sequential(
            nn.Linear(256 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, Config.LATENT_DIM),
        )

        # --- Head ---
        self.head = nn.Linear(Config.LATENT_DIM, 2)

        # Cite Lesson 00118: Zero-Initialization Fails in Dual-Stream Latent Fusion
        # We use standard initialization (PyTorch default)

    def forward(self, img, tabular):
        # Stream A: Linear Autoregressive
        # tabular[:, :2] corresponds to [Base_FVC, Rel_Time]
        linear_input = tabular[:, :2]
        linear_latent = self.linear_stream(linear_input)

        # Stream B: Deep Interaction
        vis_feats = self.backbone(img)
        vis_proj = self.vis_projector(vis_feats)

        tab_proj = self.tab_projector(tabular)

        # Concatenate and Fuse (Cite Lesson 00032: Non-Linear Mixing)
        deep_cat = torch.cat([vis_proj, tab_proj], dim=1)
        deep_latent = self.deep_fusion(deep_cat)

        # Latent Summation (Cite Lesson 00052)
        fused = linear_latent + deep_latent

        preds = self.head(fused)

        mu = preds[:, 0]
        sigma = F.softplus(preds[:, 1]) + 1e-6

        return mu, sigma
