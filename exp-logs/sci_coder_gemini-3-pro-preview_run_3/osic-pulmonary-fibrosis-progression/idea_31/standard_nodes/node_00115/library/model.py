import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class EfficientNetEncoder(nn.Module):
    """
    EfficientNet-B2 backbone with specific fine-tuning strategy.
    Freezes lower layers and unfreezes the top two convolutional stages
    and the head to allow domain adaptation while preserving robust low-level features.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained EfficientNet-B2
        # num_classes=0 removes the classifier, returning the pooled feature vector
        # global_pool='avg' ensures we get a flat vector (Batch, Num_Features)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

        # Project high-dim features (1408 for B2) to compact latent space
        self.projection = nn.Linear(self.backbone.num_features, Config.PROJECTION_DIM)

        # --- Fine-Tuning Strategy ---
        # 1. Freeze everything initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the Head (Conv + BN before pooling)
        # In timm efficientnet, this is usually conv_head and bn2
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Unfreeze the top two convolutional stages (last 2 blocks)
        # self.backbone.blocks is a nn.Sequential of blocks
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze the last 2 blocks
            for i in range(num_blocks - 2, num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

    def forward(self, x):
        # x: (B, 3, H, W)
        # features: (B, 1408)
        features = self.backbone(x)
        # projected: (B, 64)
        return self.projection(features)


class ZIMARNet(nn.Module):
    """
    Zero-Initialized Metric-Aligned Residual Network (ZIMAR-Net).

    Features:
    - Parallel Dual-Stream Latent Summation.
    - Stream A: Over-Parameterized Clinical Anchor (Robust Prior).
    - Stream B: Visual Interaction Stream (Perturbative Correction).
    - Zero-Initialization: Stream B starts at 0 contribution.
    """

    def __init__(self):
        super().__init__()

        # 1. Image Branch
        self.encoder = EfficientNetEncoder()

        # 2. Stream A: Clinical Anchor (Linear Residual Stream)
        # Input: [Baseline_FVC, Time, Age, Sex, Smoking] (Dim=5)
        # Cite Lesson 00052: Use a linear residual stream for strong autoregressive signals.
        # Cite Lesson 00060: Over-parameterize the linear stream (Project to latent dim).
        self.clinical_linear = nn.Linear(5, Config.MLP_OUT_DIM)

        # 3. Stream B: Visual Interaction
        # Input: Image Projection (64) + Clinical (5) = 69
        # Early Fusion allows learning cross-modal interactions
        self.visual_mlp = nn.Sequential(
            nn.Linear(Config.PROJECTION_DIM + 5, Config.MLP_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.MLP_HIDDEN_DIM, Config.MLP_OUT_DIM),
        )

        # 4. Zero Initialization for Stream B
        self._init_zero_residual()

        # 5. Shared Head
        # Projects fused latent (64) to Mu and Sigma (2)
        self.head = nn.Linear(Config.MLP_OUT_DIM, 2)

    def _init_zero_residual(self):
        """
        Explicitly initializes the weights and biases of the final layer
        of the Visual Stream to zero. This ensures that at Epoch 0, the
        visual stream contributes exactly 0 to the latent representation,
        forcing the model to start as a pure Clinical Anchor.
        """
        # The last layer is at index 2 in the visual_mlp sequential block
        last_layer = self.visual_mlp[2]
        if isinstance(last_layer, nn.Linear):
            nn.init.zeros_(last_layer.weight)
            nn.init.zeros_(last_layer.bias)

    def forward(self, image, clinical):
        """
        Args:
            image: (B, 3, H, W)
            clinical: (B, 5) -> [Baseline_FVC, Time, Age, Sex, Smoking]
        """
        # --- Image Branch ---
        img_embed = self.encoder(image)  # (B, 64)

        # --- Stream A (Clinical Anchor) ---
        stream_a_out = self.clinical_linear(clinical)  # (B, 64)

        # --- Stream B (Visual Interaction) ---
        # Early Fusion: Concatenate Image Embedding and Clinical Data
        stream_b_in = torch.cat([img_embed, clinical], dim=1)  # (B, 69)
        stream_b_out = self.visual_mlp(stream_b_in)  # (B, 64)

        # --- Latent Fusion (Residual Summation) ---
        # H_final = A + B
        # At epoch 0, B is 0, so H_final = A (Pure Clinical Model)
        h_final = stream_a_out + stream_b_out

        # --- Prediction Head ---
        logits = self.head(h_final)  # (B, 2)

        mu = logits[:, 0]
        # Enforce positivity for sigma using Softplus + Epsilon
        # We do not enforce the 70ml clip here (done in metric/post-processing)
        sigma = F.softplus(logits[:, 1]) + 1e-6

        return mu, sigma
