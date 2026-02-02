import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class RCRFNet(nn.Module):
    """
    Robust Clinical-Residual Fusion Network (RCRF-Net).

    Architecture:
    1. Image Branch: EfficientNet-B2 (Top 2 stages unfrozen) -> GAP -> Linear(64)
    2. Stream A (Clinical Anchor): MLP(Clinical -> 128 -> 64)
    3. Stream B (Visual Interaction): MLP([Image, Clinical] -> 128 -> 64)
    4. Fusion: Stream A + Stream B (Residual connection concept)
    5. Heads:
       - Mean: Linear(64 -> 1)
       - Uncertainty: Linear(64 + |t_rel| -> 1) -> Softplus
    """

    def __init__(self):
        super(RCRFNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Branch (Backbone)
        # ---------------------------------------------------------------------
        # Load EfficientNet-B2, remove classification head (num_classes=0)
        # This returns the pooled feature vector (1408 dim for B2)
        self.backbone = timm.create_model(
            Config.EFFNET_ARCH, pretrained=True, num_classes=0, global_pool="avg"
        )

        # Feature dimension for EfficientNet-B2 is 1408
        self.img_feature_dim = self.backbone.num_features

        # Projection to Latent Dim (64)
        self.img_projector = nn.Linear(self.img_feature_dim, Config.LATENT_DIM)

        # ---------------------------------------------------------------------
        # Freezing Logic (Fine-tune top 2 stages)
        # ---------------------------------------------------------------------
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the Head and BatchNorm
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        # Unfreeze the last N blocks (Top stages)
        # EfficientNet usually has 7 blocks. We unfreeze the last 2.
        num_blocks = len(self.backbone.blocks)
        trainable_blocks = [num_blocks - 1, num_blocks - 2]

        for i in trainable_blocks:
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # ---------------------------------------------------------------------
        # 2. Stream A: Over-Parameterized Clinical Anchor
        # ---------------------------------------------------------------------
        # Input: Clinical Features (7)
        # Arch: Linear (Direct Projection) - Cite solution_lesson_node_00052, solution_lesson_node_00060
        # We use a single linear layer to project clinical features to latent space.
        # This preserves the strong linear signal of Baseline FVC without non-linear distortion,
        # acting as a robust linear residual stream.
        self.stream_a = nn.Linear(Config.CLINICAL_INPUT_DIM, Config.LATENT_DIM)

        # ---------------------------------------------------------------------
        # 3. Stream B: Visual Interaction Stream
        # ---------------------------------------------------------------------
        # Input: Image Projection (64) + Clinical Features (7)
        # Arch: Linear -> ReLU -> Linear
        input_dim_b = Config.LATENT_DIM + Config.CLINICAL_INPUT_DIM
        self.stream_b = nn.Sequential(
            nn.Linear(input_dim_b, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # ---------------------------------------------------------------------
        # 4. Heads
        # ---------------------------------------------------------------------
        # Mean Head: Projects Latent Fusion (64) -> 1
        self.mean_head = nn.Linear(Config.LATENT_DIM, 1)

        # Uncertainty Head: Projects [Latent Fusion (64) + |t_rel| (1)] -> 1
        # This is the "Time-Shortcut" structural innovation
        self.uncertainty_head = nn.Linear(Config.LATENT_DIM + 1, 1)

    def forward(self, img, clinical, t_rel):
        """
        Args:
            img: Tensor (B, 3, 260, 260)
            clinical: Tensor (B, 7) - [BaseFVC, Time, Age, Sex, Smoke1, Smoke2, Smoke3]
            t_rel: Tensor (B, 1) or (B,) - Scaled relative time
        """
        # Ensure t_rel is (B, 1)
        if t_rel.dim() == 1:
            t_rel = t_rel.unsqueeze(1)

        # ---------------------------------------------------------------------
        # Image Processing
        # ---------------------------------------------------------------------
        # Extract features: (B, 1408)
        img_feats = self.backbone(img)

        # Project: (B, 64)
        img_latents = self.img_projector(img_feats)

        # ---------------------------------------------------------------------
        # Stream A (Clinical Anchor)
        # ---------------------------------------------------------------------
        # (B, 64)
        out_a = self.stream_a(clinical)

        # ---------------------------------------------------------------------
        # Stream B (Visual Interaction)
        # ---------------------------------------------------------------------
        # Concatenate Image Latents + Clinical: (B, 64 + 7)
        in_b = torch.cat([img_latents, clinical], dim=1)
        out_b = self.stream_b(in_b)

        # ---------------------------------------------------------------------
        # Fusion (Residual Summation)
        # ---------------------------------------------------------------------
        # (B, 64)
        h_final = out_a + out_b

        # ---------------------------------------------------------------------
        # Heads
        # ---------------------------------------------------------------------
        # 1. Mean Prediction
        mu = self.mean_head(h_final)

        # 2. Uncertainty Prediction (Time-Shortcut)
        # Concatenate |t_rel| to the latent representation
        # This allows the model to explicitly scale uncertainty with time distance
        abs_time = torch.abs(t_rel)
        sigma_input = torch.cat([h_final, abs_time], dim=1)

        raw_sigma = self.uncertainty_head(sigma_input)

        # Apply Softplus to ensure positivity + epsilon for stability
        # We do NOT clip to 70 here (that happens in metric/post-processing)
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma
