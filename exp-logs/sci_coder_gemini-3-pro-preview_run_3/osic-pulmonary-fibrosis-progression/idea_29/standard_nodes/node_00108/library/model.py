import torch
import torch.nn as nn
import timm
from library.config import Config


class MAOPDSNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Refined based on Lesson 52 and 60.

    Architecture:
    1. Image Branch: EfficientNet-B2 (Top 2 stages unfrozen) -> Global Pool -> Projection
    2. Stream A (Deep Interaction): MLP(Concat(Image, All_Tabular)) -> Latent
    3. Stream B (Linear Residual): Linear(Baseline_FVC, Time) -> Latent
    4. Fusion: Element-wise Sum (Stream A + Stream B)
    5. Head: Shared Linear Layer -> mu, sigma
    """

    def __init__(self):
        super(MAOPDSNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch
        # ---------------------------------------------------------------------
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, in_chans=Config.NUM_SLICES
        )
        self.num_img_features = self.backbone.num_features

        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze Head components
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Unfreeze last 2 stages of blocks
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            for i in range(max(0, num_blocks - 2), num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Image Projection Layer
        self.img_projector = nn.Linear(self.num_img_features, Config.IMG_EMBED_DIM)

        # ---------------------------------------------------------------------
        # 2. Stream A: Deep Interaction Stream (Image + All Tabular)
        # ---------------------------------------------------------------------
        # Input: Concatenation of [Image Projection, Clinical Features]
        input_dim_a = Config.IMG_EMBED_DIM + Config.CLINICAL_INPUT_DIM

        self.stream_a = nn.Sequential(
            nn.Linear(input_dim_a, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.IMG_EMBED_DIM),
        )

        # ---------------------------------------------------------------------
        # 3. Stream B: Linear Residual Stream (Baseline + Time)
        # ---------------------------------------------------------------------
        # Input: Baseline FVC and Time only (2 features)
        # We project to latent dim to avoid bottlenecking (Cite solution_lesson_node_00060)
        # but keep it strictly linear (Cite solution_lesson_node_00052)
        self.stream_b = nn.Linear(2, Config.IMG_EMBED_DIM)

        # ---------------------------------------------------------------------
        # 4. Shared Head
        # ---------------------------------------------------------------------
        # Input: Latent dim (64)
        self.head = nn.Linear(Config.IMG_EMBED_DIM, Config.OUTPUT_DIM)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor of shape (B, 3, 260, 260)
            tabular: Tensor of shape (B, 5) -> [BaseFVC, Time, Age, Sex, Smoke]
        """
        # --- Image Branch ---
        x_img = self.backbone.forward_features(image)
        x_img = torch.mean(x_img, dim=[2, 3])
        img_embed = self.img_projector(x_img)  # (B, 64)

        # --- Stream A (Deep Interaction) ---
        # Concatenate Image Embedding with ALL tabular features
        combined_input = torch.cat([img_embed, tabular], dim=1)
        out_a = self.stream_a(combined_input)  # (B, 64)

        # --- Stream B (Linear Residual) ---
        # Extract dominant features: Baseline FVC (idx 0) and Time (idx 1)
        linear_input = tabular[:, 0:2]
        out_b = self.stream_b(linear_input)  # (B, 64)

        # --- Latent Fusion ---
        # Residual Summation: Deep Correction + Linear Trend
        h_final = out_a + out_b  # (B, 64)

        # --- Head ---
        output = self.head(h_final)  # (B, 2)

        return output
