import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageEncoder(nn.Module):
    """
    Encodes 3-slice CT scans using EfficientNet-B2.
    Unfreezes the top 2 stages for fine-tuning.
    """

    def __init__(self):
        super().__init__()
        # Load EfficientNet B2
        # in_chans=3 corresponds to the 3 selected slices stacked as channels
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            in_chans=Config.IN_CHANNELS,
            num_classes=0,  # Remove classifier
            global_pool="",  # We handle pooling manually
        )

        # 1. Freeze entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the top 2 blocks (stages)
        # EfficientNet implementation in timm stores stages in .blocks
        # We unfreeze the last 2 blocks for high-level feature adaptation
        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        # 3. Unfreeze the final conv head and batch norm
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Projection Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(self.backbone.num_features, Config.IMG_EMBED_DIM)

    def forward(self, x):
        # x shape: (Batch, 3, 260, 260)
        features = self.backbone.forward_features(x)  # (Batch, C, H, W)
        pooled = self.global_pool(features).flatten(1)  # (Batch, C)
        embed = self.project(pooled)  # (Batch, 64)
        return embed


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network.
    Cite {lesson_id}: solution_lesson_node_00052
    Combines a Linear Autoregressive Stream with a Deep Visual Interaction Stream via summation in latent space.
    """

    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()

        # Dimensions
        input_dim = Config.CLINICAL_INPUT_DIM  # 6
        img_dim = Config.IMG_EMBED_DIM  # 64
        hidden_dim = Config.CLINICAL_HIDDEN_DIM  # 128
        out_dim = Config.CLINICAL_OUT_DIM  # 64

        # Stream A: Linear Autoregressive Residual
        # Cite {lesson_id}: solution_lesson_node_00060 (Over-parameterization)
        # Input: [Base_FVC (Scaled), Rel_Week] -> 2 features
        self.stream_a = nn.Linear(2, out_dim)

        # Stream B: Deep Visual Interaction
        # Input: Image Embed (64) + All Clinical Features (6) = 70
        self.stream_b = nn.Sequential(
            nn.Linear(input_dim + img_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Shared Heads
        # Cite {lesson_id}: solution_lesson_node_00055 (Shared path for uncertainty)
        self.head_mu = nn.Linear(out_dim, 1)
        self.head_sigma = nn.Linear(out_dim, 1)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (B, 3, H, W)
            tabular: Tensor (B, 6) -> [Base_FVC, Base_Percent, Rel_Week, Age, Sex, Smoke]
        """
        # 1. Extract Image Embeddings
        img_embed = self.image_encoder(image)  # (B, 64)

        # 2. Stream A: Autoregressive
        # Select Base_FVC (idx 0) and Rel_Week (idx 2)
        ar_input = tabular[:, [0, 2]]  # (B, 2)
        out_a = self.stream_a(ar_input)  # (B, 64)

        # 3. Stream B: Visual Interaction
        combined_input = torch.cat([img_embed, tabular], dim=1)  # (B, 70)
        out_b = self.stream_b(combined_input)  # (B, 64)

        # 4. Latent Fusion (Summation)
        # The linear trend (A) is corrected by the deep residual (B)
        h_final = out_a + out_b  # (B, 64)

        # 5. Prediction Heads
        mu = self.head_mu(h_final)
        raw_sigma = self.head_sigma(h_final)

        # Enforce positivity
        sigma = F.softplus(raw_sigma) + Config.SIGMA_MIN_TRAIN

        return mu, sigma
