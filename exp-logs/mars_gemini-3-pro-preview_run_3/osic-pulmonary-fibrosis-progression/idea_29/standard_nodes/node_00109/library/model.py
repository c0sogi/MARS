import torch
import torch.nn as nn
import timm
from library.config import Config


class MAOPDSNet(nn.Module):
    """
    Metric-Aligned Over-Parameterized Dual-Stream Network (MAOP-DS Net).

    Architecture:
    1. Image Branch: EfficientNet-B2 (Top 2 stages unfrozen) -> Global Pool -> Projection
    2. Stream A: Clinical Anchor MLP (Input -> Latent)
    3. Stream B: Visual Interaction MLP (Concat(Image, Input) -> Latent)
    4. Fusion: Element-wise Sum (Stream A + Stream B)
    5. Head: Shared Linear Layer -> mu, sigma
    """

    def __init__(self):
        super(MAOPDSNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch
        # ---------------------------------------------------------------------
        # Load backbone
        # num_classes=0 removes the classifier, returning the pooled features or features before pooling
        # We use forward_features so we handle pooling manually or let timm handle it if configured
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, in_chans=Config.NUM_SLICES
        )

        # Get the number of features output by the backbone
        # For EfficientNet-B2, this is typically 1408
        self.num_img_features = self.backbone.num_features

        # --- Freezing Logic ---
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top layers (Top 2 Convolutional Stages)
        # In timm EfficientNet, the structure is typically:
        # conv_stem -> bn1 -> blocks (Sequential) -> conv_head -> bn2

        # Unfreeze Head components
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 stages of blocks
        # blocks is a nn.Sequential containing the stages
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(max(0, num_blocks - 2), num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Image Projection Layer
        self.img_projector = nn.Linear(self.num_img_features, Config.IMG_EMBED_DIM)

        # ---------------------------------------------------------------------
        # 2. Stream A: Over-Parameterized Linear Residual
        # ---------------------------------------------------------------------
        # Input: Clinical features only
        # Architecture: Single Linear Layer (Over-parameterized projection)
        # Cite Lesson 52: Dual-Stream Point-Wise Residuals (Linear Stream)
        # Cite Lesson 60: Over-Parameterization of Linear Baselines
        self.stream_a = nn.Linear(Config.CLINICAL_INPUT_DIM, Config.IMG_EMBED_DIM)

        # ---------------------------------------------------------------------
        # 3. Stream B: Visual Interaction Stream
        # ---------------------------------------------------------------------
        # Input: Concatenation of [Image Projection, Clinical Features]
        input_dim_b = Config.IMG_EMBED_DIM + Config.CLINICAL_INPUT_DIM

        self.stream_b = nn.Sequential(
            nn.Linear(input_dim_b, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(
                Config.HIDDEN_DIM, Config.IMG_EMBED_DIM
            ),  # Project to latent dim (64)
        )

        # ---------------------------------------------------------------------
        # 4. Shared Head
        # ---------------------------------------------------------------------
        # Input: Latent dim (64) (Result of Summation)
        # Output: 2 (mu, sigma)
        self.head = nn.Linear(Config.IMG_EMBED_DIM, Config.OUTPUT_DIM)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor of shape (B, 3, 260, 260)
            tabular: Tensor of shape (B, 5) -> [BaseFVC, Time, Age, Sex, Smoke]
        """
        # --- Image Branch ---
        # Extract features: (B, C, H, W)
        x_img = self.backbone.forward_features(image)

        # Global Average Pooling: (B, C)
        # EfficientNet features are (B, 1408, H, W)
        x_img = torch.mean(x_img, dim=[2, 3])

        # Project to embedding dimension
        img_embed = self.img_projector(x_img)  # (B, 64)

        # --- Stream A (Clinical Anchor) ---
        # Learns the expected trajectory based purely on clinical data
        out_a = self.stream_a(tabular)  # (B, 64)

        # --- Stream B (Visual Interaction) ---
        # Learns residuals based on image + clinical context
        # Early Fusion
        combined_input = torch.cat([img_embed, tabular], dim=1)  # (B, 64 + 5)
        out_b = self.stream_b(combined_input)  # (B, 64)

        # --- Latent Fusion ---
        # Residual Summation
        h_final = out_a + out_b  # (B, 64)

        # --- Head ---
        output = self.head(h_final)  # (B, 2)

        return output
