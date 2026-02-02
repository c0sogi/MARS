import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class PriorStream(nn.Module):
    """
    Stream A: Linear Residual (Cite Lesson 00052, 00060).
    Projects dominant autoregressive features (Base_FVC, Time) into latent space.
    """

    def __init__(self, input_dim=2, output_dim=64):
        super(PriorStream, self).__init__()
        # Over-parameterize linear baseline (Cite Lesson 00060)
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.net(x)


class InteractionStream(nn.Module):
    """
    Stream B: Deep Interaction (Cite Lesson 00052).
    Fuses Image and Clinical Metadata to learn non-linear corrections.
    """

    def __init__(self, img_dim, clin_dim=9, latent_dim=64):
        super(InteractionStream, self).__init__()
        # Pre-fusion projection (Cite Lesson 00008)
        self.img_proj = nn.Linear(img_dim, latent_dim)
        self.clin_proj = nn.Linear(clin_dim, latent_dim)

        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, img_feat, clin_data):
        i = self.img_proj(img_feat)
        c = self.clin_proj(clin_data)
        # Fuse
        combined = torch.cat([i, c], dim=1)
        return self.fusion(combined)


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPR-Net).
    Combines a strong linear prior stream with a deep interaction stream.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch (Backbone)
        # ---------------------------------------------------------------------
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool=""
        )
        self.num_features = self.backbone.num_features

        # Freezing Logic (Cite Lesson 00027)
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
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ---------------------------------------------------------------------
        # 2. Streams
        # ---------------------------------------------------------------------
        # Prior Stream: Base_FVC, Rel_Time
        self.prior_stream = PriorStream(input_dim=2, output_dim=Config.LATENT_DIM)

        # Interaction Stream: Image + All Clinical
        self.interaction_stream = InteractionStream(
            img_dim=self.num_features, clin_dim=9, latent_dim=Config.LATENT_DIM
        )

        # ---------------------------------------------------------------------
        # 3. Prediction Head
        # ---------------------------------------------------------------------
        # No Dropout in regression head (Cite Lesson 00007)
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, img, clin_data):
        # Image Features
        feat = self.backbone.forward_features(img)
        feat = self.global_pool(feat).flatten(1)

        # Prior Stream Inputs: Base_FVC (idx 0), Rel_Time (idx 3)
        prior_in = clin_data[:, [0, 3]]
        h_prior = self.prior_stream(prior_in)

        # Interaction Stream
        h_deep = self.interaction_stream(feat, clin_data)

        # Fusion (Summation) - Cite Lesson 00052
        h_final = h_prior + h_deep

        return self.head(h_final)
