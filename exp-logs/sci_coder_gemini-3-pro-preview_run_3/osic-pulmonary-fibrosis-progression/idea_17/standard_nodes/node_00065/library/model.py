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
        # 2. Stream A: Linear Residual (Base_FVC, Time)
        # ---------------------------------------------------------------------
        # Cite Lesson 00052: Use a linear residual stream for dominant autoregressive features.
        # Cite Lesson 00060: Over-parameterize the linear stream in latent space.
        # Input: [Base_FVC, t_rel] -> 2 features
        self.linear_stream = nn.Linear(2, Config.EMBED_DIM)

        # ---------------------------------------------------------------------
        # 3. Stream B: Deep Interaction (Image + Tabular)
        # ---------------------------------------------------------------------
        # Tabular Encoder for the deep stream (All features)
        self.tabular_input_dim = 7
        self.tabular_encoder = nn.Sequential(
            nn.Linear(self.tabular_input_dim, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM),
        )

        # Deep Fusion: Concat(Image_Embed, Tab_Embed) -> 128 + 128 = 256
        self.deep_fusion = nn.Sequential(
            nn.Linear(Config.EMBED_DIM * 2, Config.EMBED_DIM),
            nn.ReLU(),
            nn.Linear(Config.EMBED_DIM, Config.EMBED_DIM),
        )

        # ---------------------------------------------------------------------
        # 4. Head
        # ---------------------------------------------------------------------
        # Projects H_final to [mu, raw_sigma]
        # Cite Lesson 00055: Do not isolate strong domain priors from the uncertainty head.
        # By summing streams before the head, both contribute to mu and sigma.
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
        features = self.backbone.forward_features(images)
        features = self.global_pool(features).flatten(1)
        img_embed = self.img_project(features)

        # --- Stream A: Linear Residual ---
        # Projects Base_FVC and Time (indices 0, 1) linearly to latent space
        h_linear = self.linear_stream(tabular[:, :2])

        # --- Stream B: Deep Interaction ---
        # Encode all tabular features for the non-linear path
        h_tab = self.tabular_encoder(tabular)

        # Concatenate Image and Tabular embeddings
        combined = torch.cat([img_embed, h_tab], dim=1)

        # Compute deep residual features
        h_deep = self.deep_fusion(combined)

        # --- Fusion ---
        # Sum linear trend and deep residual in latent space
        h_final = h_linear + h_deep

        # --- Output Head ---
        out = self.head(h_final)

        return out
