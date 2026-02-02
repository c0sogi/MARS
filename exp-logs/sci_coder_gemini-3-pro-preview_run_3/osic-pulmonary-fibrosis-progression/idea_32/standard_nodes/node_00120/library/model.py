import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Fuses a Linear Residual Stream (for strong autoregressive priors) with a
    Deep Interaction Stream (for image-based corrections) in a latent space.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # Stream A: Linear Residual Stream
        # Projects dominant autoregressive features (BaseFVC, Time) to latent space.
        # Cite Lesson 00052: Use linear residual stream for baseline dependency.
        # Cite Lesson 00060: Over-parameterize linear baseline in latent space (64 dim).
        self.linear_stream = nn.Linear(2, 64)

        # Stream B: Deep Interaction Stream
        # Extracts features from Image and Tabular data.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Fine-Tuning Strategy
        # Cite Lesson 00027: Differential Learning Rates
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

        self.img_feature_dim = self.backbone.num_features

        # Deep Stream Projection
        # Concatenates Image + Tabular (5) -> MLP -> Latent (64)
        self.deep_stream = nn.Sequential(
            nn.Linear(self.img_feature_dim + 5, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(256, 64),
        )

        # Shared Head
        # Fused Latent (64) -> Output (2)
        # Cite Lesson 00055: Shared path for uncertainty
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, 2))

        # Cite Lesson 00118: Standard initialization (no zero init) for latent fusion.

    def forward(self, img, tab):
        # Stream A: Linear (BaseFVC at idx 0, Time at idx 1)
        # Cite Lesson 00052: Linear mapping of dominant autoregressive features
        lin_feat = self.linear_stream(tab[:, :2])

        # Stream B: Deep
        img_feat = self.backbone(img)
        deep_in = torch.cat([img_feat, tab], dim=1)
        deep_feat = self.deep_stream(deep_in)

        # Latent Fusion (Summation)
        # Cite Lesson 00060: Summing in latent space
        fused = lin_feat + deep_feat

        # Prediction
        out = self.head(fused)

        mu = out[:, 0]
        # Cite Lesson 00066: Metric-aligned loss expects sigma, use softplus
        sigma = F.softplus(out[:, 1]) + 1e-6

        return mu, sigma
