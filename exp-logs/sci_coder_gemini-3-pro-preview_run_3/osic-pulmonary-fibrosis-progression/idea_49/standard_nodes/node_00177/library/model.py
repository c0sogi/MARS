import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageBranch(nn.Module):
    """
    Fine-Tuned Content-Adaptive 2.5D Image Branch.
    Extracts features from CT scans and projects them to a low-dimensional bottleneck.
    """

    def __init__(self):
        super(ImageBranch, self).__init__()
        # Load EfficientNet-B2
        # num_classes=0 removes the classifier and returns pooled features (1408 dim for B2)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # --- Fine-Tuning Strategy ---
        # 1. Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the top two convolutional stages
        # EfficientNet-B2 structure in timm typically has 'blocks' 0 through 6.
        # We unfreeze the last two blocks (5 and 6) and the final conv head/bn.
        if hasattr(self.backbone, "blocks"):
            for param in self.backbone.blocks[5].parameters():
                param.requires_grad = True
            for param in self.backbone.blocks[6].parameters():
                param.requires_grad = True

        # Unfreeze conv_head and bn2 (final layers before pooling)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Bottleneck Projection
        # Projects high-dimensional image noise (1408) to structural embedding (64)
        self.projection = nn.Linear(Config.BACKBONE_OUT_DIM, Config.PROJECTION_DIM)

    def forward(self, x):
        # x: (B, 3, H, W)
        features = self.backbone(x)  # (B, 1408)
        projected = self.projection(features)  # (B, 64)
        return projected


class SCARNet(nn.Module):
    """
    Refined Architecture: Dual-Stream Point-Wise Residual Network (DSPRNet).
    Fuses a wide linear stream and a deep interaction stream in latent space.
    """

    def __init__(self):
        super(SCARNet, self).__init__()

        # 1. Image Branch (Projected to 64 dim)
        self.image_branch = ImageBranch()

        # 2. Dimensions
        self.clinical_dim = 5
        self.latent_dim = 64
        self.deep_hidden_dim = 128

        # 3. Stream Linear (Wide): Projects Tabular to Latent Space
        # Cite solution_lesson_node_00060: Over-parameterize linear baseline
        self.stream_linear = nn.Linear(self.clinical_dim, self.latent_dim)

        # 4. Stream Deep (Interaction): Image + Tabular -> Latent
        # Cite solution_lesson_node_00139: Context visibility (Tabular in Deep stream)
        # Cite solution_lesson_node_00146: Dimensionality balance (Image projected to 64 before concat)
        self.deep_input_dim = Config.PROJECTION_DIM + self.clinical_dim
        self.stream_deep = nn.Sequential(
            nn.Linear(self.deep_input_dim, self.deep_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.deep_hidden_dim, self.latent_dim),
        )

        # 5. Shared Head
        # Cite solution_lesson_node_00052: Summed in latent space
        self.head = nn.Sequential(
            nn.ReLU(),  # Activation on fused latent
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, Config.OUTPUT_DIM),
        )

        self.sigma_floor = Config.SIGMA_FLOOR_STD

    def forward(self, images, tabular):
        # Image Features (B, 64)
        img_feat = self.image_branch(images)

        # Stream Linear (B, 64)
        linear_feat = self.stream_linear(tabular)

        # Stream Deep (B, 64)
        deep_in = torch.cat([img_feat, tabular], dim=1)
        deep_feat = self.stream_deep(deep_in)

        # Latent Fusion (Sum)
        fused = linear_feat + deep_feat

        # Output
        out = self.head(fused)

        mu = out[:, 0:1]
        sigma_raw = out[:, 1:2]

        sigma = F.softplus(sigma_raw) + self.sigma_floor
        return torch.cat([mu, sigma], dim=1)
