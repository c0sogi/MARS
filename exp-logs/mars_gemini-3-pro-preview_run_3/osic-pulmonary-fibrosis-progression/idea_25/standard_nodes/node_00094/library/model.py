import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBranch(nn.Module):
    """
    Fine-Tuned Content-Adaptive 2.5D Image Branch.
    Backbone: EfficientNet-B2
    """

    def __init__(self):
        super(VisualBranch, self).__init__()

        # Load backbone with pretrained weights
        # in_chans=3 corresponds to the 3 slices we stack
        # num_classes=0 removes the classifier and returns the pooled features directly
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
            in_chans=Config.SLICES_PER_PATIENT,
        )

        # Get feature dimension (EfficientNet-B2 usually 1408)
        self.in_features = self.backbone.num_features

        # Projection Head: Projects to 64 dimensions
        self.projection = nn.Linear(self.in_features, Config.IMG_EMBED_DIM)

        # Freeze all backbone layers initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top two convolutional stages
        # In timm EfficientNets, the structure is typically:
        # conv_stem -> bn1 -> blocks -> conv_head -> bn2
        # 'blocks' is a nn.Sequential containing the MBConv blocks.

        # 1. Unfreeze Head components (conv_head, bn2)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 2. Unfreeze the last two blocks of the 'blocks' container
        # This corresponds to the deepest architectural stages
        if hasattr(self.backbone, "blocks"):
            num_blocks = len(self.backbone.blocks)
            # Unfreeze last 2 blocks
            for i in range(max(0, num_blocks - 2), num_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # Ensure projection layer is trainable
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x shape: (Batch, 3, 260, 260)
        # Extract features
        features = self.backbone(x)  # (Batch, num_features)
        # Project
        embedding = self.projection(features)  # (Batch, 64)
        return embedding


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (Cite solution_lesson_node_00052).
    Combines a deep interaction stream (Image + Tabular) with a linear residual stream (Baseline + Time).
    """

    def __init__(self):
        super(DSPRNet, self).__init__()
        self.visual_branch = VisualBranch()

        # Stream A: Deep Interaction (Image + Full Tabular)
        # Tabular encoder
        self.tab_encoder = nn.Linear(6, Config.LATENT_DIM)  # 6 -> 64

        # Interaction MLP
        self.deep_mlp = nn.Sequential(
            nn.Linear(Config.IMG_EMBED_DIM + Config.LATENT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, Config.LATENT_DIM),
        )

        # Stream B: Linear Residual (Baseline FVC + Weeks)
        # Projected to latent space (Cite solution_lesson_node_00060)
        self.linear_stream = nn.Linear(2, Config.LATENT_DIM)

        # Shared Head
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        # 1. Visual Features
        img_embed = self.visual_branch(image)  # (B, 64)

        # 2. Stream A: Deep Interaction
        tab_embed = F.relu(self.tab_encoder(tabular))  # (B, 64)
        deep_in = torch.cat([img_embed, tab_embed], dim=1)
        deep_latent = self.deep_mlp(deep_in)  # (B, 64)

        # 3. Stream B: Linear Residual
        # Inputs: Baseline_FVC (idx 0) and Relative_Weeks (idx 5)
        linear_input = tabular[:, [0, 5]]
        linear_latent = self.linear_stream(linear_input)  # (B, 64)

        # 4. Fusion (Summation in latent space)
        h_final = deep_latent + linear_latent

        # 5. Prediction Head
        out = self.head(h_final)
        mu = out[:, 0]
        sigma = F.softplus(out[:, 1]) + 1e-6

        return mu, sigma
