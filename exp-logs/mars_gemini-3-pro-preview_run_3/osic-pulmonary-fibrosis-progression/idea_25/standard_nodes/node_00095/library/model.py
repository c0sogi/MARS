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


class LinearStream(nn.Module):
    """
    Stream A: Linear Residual Stream.
    Processes dominant autoregressive features (Baseline FVC, Time).
    Over-parameterized to latent dimension to avoid bottlenecks (Cite solution_lesson_node_00060).
    """

    def __init__(self):
        super(LinearStream, self).__init__()
        # Inputs: Baseline_FVC_Scaled, Relative_Weeks_Scaled
        input_dim = 2
        # Project to latent dim directly (Linear mapping)
        self.net = nn.Linear(input_dim, Config.LATENT_DIM)

    def forward(self, x):
        return self.net(x)


class DeepStream(nn.Module):
    """
    Stream B: Deep Interaction Stream.
    Fuses Image Embeddings with all Tabular features to learn non-linear corrections.
    """

    def __init__(self):
        super(DeepStream, self).__init__()
        # Input: Image Projection (64) + All Tabular (6)
        input_dim = Config.IMG_EMBED_DIM + 6

        self.net = nn.Sequential(
            nn.Linear(input_dim, Config.FUSION_HIDDEN_DIM),  # 128
            nn.ReLU(),
            nn.Linear(Config.FUSION_HIDDEN_DIM, Config.LATENT_DIM),  # 64
        )

    def forward(self, img_embed, tabular):
        combined = torch.cat([img_embed, tabular], dim=1)
        return self.net(combined)


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Implements architecture from solution_lesson_node_00052.
    """

    def __init__(self):
        super(DSPRNet, self).__init__()
        self.visual_branch = VisualBranch()
        self.linear_stream = LinearStream()
        self.deep_stream = DeepStream()

        # Shared Head: Projects H_final to mu and sigma
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (Batch, 3, H, W)
            tabular: Tensor (Batch, 6)
        Returns:
            mu: Predicted FVC (scaled)
            sigma: Predicted Uncertainty (scaled)
        """
        # 1. Get Visual Embedding
        img_embed = self.visual_branch(image)  # (B, 64)

        # 2. Linear Stream (Stream A)
        # Select Baseline_FVC (idx 0) and Relative_Weeks (idx 5)
        # Tabular structure: [BaseFVC, BasePercent, Age, Sex, Smoke, Weeks]
        linear_input = tabular[:, [0, 5]]
        h_linear = self.linear_stream(linear_input)  # (B, 64)

        # 3. Deep Stream (Stream B)
        h_deep = self.deep_stream(img_embed, tabular)  # (B, 64)

        # 4. Residual Fusion (Summation)
        h_final = h_linear + h_deep  # (B, 64)

        # 5. Prediction Head
        out = self.head(h_final)  # (B, 2)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Uncertainty Constraint
        # Apply softplus to ensure positivity, add epsilon for stability
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma
