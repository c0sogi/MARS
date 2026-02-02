import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Stream A: Linear mapping of dominant autoregressive features (Baseline + Time).
    Stream B: Deep interaction of Image + Tabular + Time.
    Fusion: Summed in latent space.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # --- Stream A: Linear Autoregressive ---
        # Input: Baseline FVC (idx 0) and Relative Time (idx 1)
        # Cite Lesson 00060: Over-parameterize linear stream to match latent dim
        self.stream_a = nn.Linear(2, Config.FUSION_LATENT_DIM)

        # --- Stream B: Deep Interaction ---
        # Backbone: EfficientNet-B2
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=3,
        )

        # Fine-Tuning: Unfreeze top layers
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

        # Deep Stream MLP
        # Input: Image Features + All Tabular Features
        # Cite Lesson 00103: Early Fusion
        input_dim = self.vis_dim + Config.TABULAR_INPUT_DIM

        # Cite Lesson 00126: No Dropout in Deep Stream
        self.stream_b_mlp = nn.Sequential(
            nn.Linear(input_dim, Config.FUSION_LATENT_DIM * 2),
            nn.ReLU(),
            nn.Linear(Config.FUSION_LATENT_DIM * 2, Config.FUSION_LATENT_DIM),
        )

        # --- Final Head ---
        self.head = nn.Linear(Config.FUSION_LATENT_DIM, 2)

    def forward(self, img, tabular):
        # --- Stream A ---
        # Select Baseline FVC (0) and Relative Time (1)
        linear_input = tabular[:, :2]
        a_latent = self.stream_a(linear_input)

        # --- Stream B ---
        vis_feats = self.backbone(img)
        # Concatenate Image + All Tabular
        b_input = torch.cat([vis_feats, tabular], dim=1)
        b_latent = self.stream_b_mlp(b_input)

        # --- Fusion ---
        # Cite Lesson 00052: Sum in latent space
        fused = a_latent + b_latent

        # --- Prediction ---
        preds = self.head(fused)
        mu = preds[:, 0]
        sigma_logit = preds[:, 1]
        sigma = F.softplus(sigma_logit) + 1e-6

        return mu, sigma
