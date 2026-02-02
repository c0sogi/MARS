import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageEncoder(nn.Module):
    """
    Fine-Tuned EfficientNet-B2 encoder for 2.5D CT slices.
    Extracts features and projects them to a compact latent space.
    """

    def __init__(self):
        super(ImageEncoder, self).__init__()

        # Load backbone with NoisyStudent weights
        # global_pool='avg' replaces the classifier with a pooling layer, returning a 1D vector
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,
            global_pool="avg",
        )

        # EfficientNet-B2 typically outputs 1408 features
        self.in_features = self.backbone.num_features

        # Projection to shared latent dimension (64-dim)
        self.projection = nn.Linear(self.in_features, Config.FEATURE_DIM)

        # Apply freezing strategy
        self._freeze_stages()

    def _freeze_stages(self):
        """
        Freezes the entire backbone, then unfreezes the top 2 convolutional blocks
        and the head to allow fine-tuning on high-level features.
        """
        # 1. Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze Head (Conv Head + BN)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Unfreeze Top 2 Blocks
        # timm EfficientNets store blocks in a Sequential container named 'blocks'
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # x shape: (Batch, 3, 260, 260)
        features = self.backbone(x)  # (Batch, 1408)
        projected = self.projection(features)  # (Batch, 64)
        return projected


class CASDSNet(nn.Module):
    """
    Constraint-Aware Standardized Dual-Stream Network.
    Combines clinical data (Anchor) and CT imaging (Residual) with metric-aligned constraints.
    """

    def __init__(self):
        super(CASDSNet, self).__init__()

        # --- Components ---
        self.image_encoder = ImageEncoder()

        # Tabular Input Dimensions:
        # [Baseline_FVC, t_rel, Age, Sex, Smoking] -> 5 features
        self.tab_dim = 5

        # Stream A: Clinical Anchor (Over-Parameterized MLP)
        # Learns the baseline linear trajectory
        self.stream_a = nn.Sequential(
            nn.Linear(self.tab_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.FEATURE_DIM),
        )

        # Stream B: Visual Residual (Context-Injected MLP)
        # Learns residuals based on Image + Clinical Context
        # Input: Image (64) + Clinical (5) = 69
        self.stream_b = nn.Sequential(
            nn.Linear(Config.FEATURE_DIM + self.tab_dim, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.FEATURE_DIM),
        )

        # Prediction Head
        # Projects fused latent state to Mean and Raw Sigma
        self.head = nn.Linear(Config.FEATURE_DIM, 2)

        # --- Constraints ---
        # Standardized floor for sigma (approx 70ml in z-score space)
        self.epsilon_std = Config.STD_MIN_SIGMA

    def forward(self, images, tabular):
        """
        Forward pass of the network.

        Args:
            images: Tensor of shape (Batch, 3, H, W)
            tabular: Tensor of shape (Batch, 5) containing normalized clinical features.

        Returns:
            mu: Predicted mean FVC (standardized)
            sigma: Predicted confidence (standardized)
        """
        # 1. Image Feature Extraction
        img_embed = self.image_encoder(images)  # (Batch, 64)

        # 2. Stream A: Clinical Anchor
        # Processes only tabular data to establish baseline
        feat_a = self.stream_a(tabular)  # (Batch, 64)

        # 3. Stream B: Visual Residual
        # Concatenates Image embedding with raw Tabular context
        combined_input = torch.cat([img_embed, tabular], dim=1)  # (Batch, 69)
        feat_b = self.stream_b(combined_input)  # (Batch, 64)

        # 4. Latent Fusion (Summation)
        # H_final = Anchor + Residual
        h_final = feat_a + feat_b  # (Batch, 64)

        # 5. Prediction Head
        outputs = self.head(h_final)  # (Batch, 2)

        mu = outputs[:, 0]
        sigma_raw = outputs[:, 1]

        # 6. Metric Constraint Enforcement
        # Enforce sigma > 70ml (in standardized space) using Softplus + Epsilon
        # This prevents the "phantom gain" optimization failure.
        sigma = F.softplus(sigma_raw) + self.epsilon_std

        return mu, sigma
