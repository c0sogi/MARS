import torch
import torch.nn as nn
import timm
from library.config import Config


class ClinicalStream(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor.
    Models the baseline disease trajectory using clinical priors.
    Cite Lesson 52: Use a linear residual stream when the target has a strong dependency on a baseline value.
    Cite Lesson 60: Over-parameterize the linear stream by projecting it into a higher-dimensional latent space.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Linear Projection: Input -> Output (No non-linearities)
        # We ignore hidden_dim here to enforce linearity and preserve the strong baseline signal
        self.net = nn.Linear(input_dim, output_dim)

        # Auxiliary Head for supervision: Output -> 2 (mu, sigma)
        self.aux_head = nn.Linear(output_dim, 2)

    def forward(self, x):
        features = self.net(x)
        aux_out = self.aux_head(features)
        return features, aux_out


class VisualStream(nn.Module):
    """
    Stream B Part 1: Feature Extraction and Projection.
    Uses EfficientNet-B2 and projects high-dim features to low-dim space.
    """

    def __init__(self, backbone_name, projection_dim):
        super().__init__()
        # Load backbone with pooled output (num_classes=0 returns global pool features)
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, in_chans=3
        )

        # Determine backbone output dimension (EfficientNet-B2 is typically 1408)
        self.num_features = self.backbone.num_features

        # Bottleneck Projection: 1408 -> 64
        self.projection = nn.Linear(self.num_features, projection_dim)

        # Freeze bottom layers, unfreeze top two stages
        self._freeze_layers()

    def _freeze_layers(self):
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top stages (blocks 5, 6 for B2) and head components
        # We look for specific strings in parameter names corresponding to the top blocks
        targets = ["blocks.5", "blocks.6", "conv_head", "bn2"]
        for name, param in self.backbone.named_parameters():
            if any(t in name for t in targets):
                param.requires_grad = True

    def forward(self, x):
        # Extract features (B, 1408)
        x = self.backbone(x)
        # Project (B, 64)
        x = self.projection(x)
        return x


class SPCRNet(nn.Module):
    """
    Supervised Projected-Context Residual Network.
    Fuses clinical anchor with visual residuals via context injection and summation.
    """

    def __init__(self):
        super().__init__()

        # Configuration
        self.tabular_input_dim = 5  # [fvc_norm, age_norm, sex, smoke, time]
        self.hidden_dim = Config.HIDDEN_DIM
        self.feature_dim = Config.PROJECTION_DIM

        # Stream A: Clinical
        self.clinical_stream = ClinicalStream(
            input_dim=self.tabular_input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.feature_dim,
        )

        # Stream B Part 1: Visual Backbone
        self.visual_stream = VisualStream(
            backbone_name=Config.BACKBONE_NAME, projection_dim=self.feature_dim
        )

        # Stream B Part 2: Residual MLP with Context Injection
        # Input: Projected Visual (64) + Raw Clinical (5) = 69
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.feature_dim + self.tabular_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )

        # Shared Head: Feature Dim -> 2 (mu, sigma)
        self.head = nn.Linear(self.feature_dim, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (B, 3, H, W)
            tabular: Tensor (B, 5)
        Returns:
            final_out: Tensor (B, 2) [mu, sigma]
            aux_out: Tensor (B, 2) [mu_aux, sigma_aux]
        """
        # 1. Clinical Stream (Anchor)
        # h_clin: (B, 64), aux_out: (B, 2)
        h_clin, aux_out = self.clinical_stream(tabular)

        # 2. Visual Stream (Projection)
        # h_vis_proj: (B, 64)
        h_vis_proj = self.visual_stream(image)

        # 3. Context Injection
        # Concatenate projected visual features with raw clinical inputs
        # (B, 64) + (B, 5) -> (B, 69)
        context_input = torch.cat([h_vis_proj, tabular], dim=1)

        # 4. Residual Learning
        # h_res: (B, 64)
        h_res = self.residual_mlp(context_input)

        # 5. Latent Fusion (Residual Connection)
        # H_final = H_clin + StreamB_output
        h_final = h_clin + h_res

        # 6. Prediction
        # final_out: (B, 2)
        final_out = self.head(h_final)

        return final_out, aux_out
