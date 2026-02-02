import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class RIDSNet(nn.Module):
    """
    Residual-Interaction Dual-Stream Network (RIDS-Net).

    Architecture:
    1. Image Branch: EfficientNet-B2 (Fine-tuned top layers) -> Projection -> 128 dim
    2. Clinical Stream (A): Tabular Input -> MLP -> 128 dim (H_clin)
    3. Interaction Stream (B): Concat(Image, H_clin) -> MLP -> 128 dim (H_resid)
    4. Fusion: H_final = H_clin + H_resid
    5. Head: H_final -> Linear -> [mu, raw_sigma]
    """

    def __init__(self):
        super(RIDSNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B2)
        # ---------------------------------------------------------------------
        # Load pretrained model
        # in_chans=3 matches the 3 slices stacked in data.py
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            in_chans=3,
            num_classes=0,  # Remove classifier
            global_pool="",  # We will handle pooling manually
        )

        # Determine number of features output by the backbone
        # For EfficientNet-B2, this is typically 1408
        num_backbone_features = self.backbone.num_features

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Image Projection Layer: Project backbone features to embedding dimension
        self.img_project = nn.Linear(num_backbone_features, Config.EMBED_DIM)

        # ---------------------------------------------------------------------
        # Freezing Logic
        # ---------------------------------------------------------------------
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top layers for adaptation
        # Unfreeze conv_head and bn2 (the final processing before classifier)
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        # Unfreeze the last two blocks of the main stage
        # EfficientNet 'blocks' is a nn.Sequential of stages.
        # We unfreeze the last 2 stages.
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # ---------------------------------------------------------------------
        # 2. Stream A: Clinical Anchor (Over-Parameterized Trajectory)
        # ---------------------------------------------------------------------
        # Input features: [Base_FVC, t_rel, Age, Sex, Smoke_0, Smoke_1, Smoke_2] -> 7 features
        self.tabular_input_dim = 7

        self.clinical_mlp = nn.Sequential(
            nn.Linear(self.tabular_input_dim, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM),
            # Output: H_clin
        )

        # ---------------------------------------------------------------------
        # 3. Stream B: Visual Interaction Stream
        # ---------------------------------------------------------------------
        # Input: Concat(Image_Embed, H_clin) -> 128 + 128 = 256
        self.interaction_input_dim = Config.EMBED_DIM * 2

        self.visual_mlp = nn.Sequential(
            nn.Linear(self.interaction_input_dim, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM),
            # Output: H_resid
        )

        # ---------------------------------------------------------------------
        # 4. Head
        # ---------------------------------------------------------------------
        # Projects H_final to [mu, raw_sigma]
        self.head = nn.Linear(Config.EMBED_DIM, 2)

    def forward(self, images, tabular):
        """
        Args:
            images: Tensor (Batch, 3, H, W)
            tabular: Tensor (Batch, 7)

        Returns:
            Tensor (Batch, 2) -> [mu, raw_sigma]
        """
        # --- Image Branch ---
        # Extract features: (B, C, H, W)
        features = self.backbone.forward_features(images)
        # Pool: (B, C, 1, 1) -> (B, C)
        features = self.global_pool(features).flatten(1)
        # Project: (B, 128)
        img_embed = self.img_project(features)

        # --- Stream A: Clinical Anchor ---
        # Compute H_clin: (B, 128)
        h_clin = self.clinical_mlp(tabular)

        # --- Stream B: Visual Interaction ---
        # Concatenate Image Embedding and Clinical Latent
        # This forces the visual stream to see the clinical context
        combined = torch.cat([img_embed, h_clin], dim=1)

        # Compute H_resid: (B, 128)
        h_resid = self.visual_mlp(combined)

        # --- Fusion ---
        # Residual connection: The visual info acts as a correction to the clinical anchor
        h_final = h_clin + h_resid

        # --- Output Head ---
        # Predict mu and raw_sigma
        out = self.head(h_final)

        return out
