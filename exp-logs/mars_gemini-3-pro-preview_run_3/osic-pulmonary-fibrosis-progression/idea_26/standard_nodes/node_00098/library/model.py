import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Cite solution_lesson_node_00052: Dual-Stream Residuals for Strong Autoregressive Signals.

    This architecture explicitly separates the modeling of the linear autoregressive trend
    (Baseline FVC + Time) from the complex non-linear corrections (Image + Context).
    """

    def __init__(self):
        super(DSPRNet, self).__init__()

        # --- Image Backbone ---
        # EfficientNet-B2 (Cite solution_lesson_node_00071: Backbone Selection)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, in_chans=3, features_only=False
        )

        # Determine backbone output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone.forward_features(dummy)
            self.backbone_dim = features.shape[1]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.img_project = nn.Linear(self.backbone_dim, Config.IMG_EMBED_DIM)

        # --- Freezing Strategy ---
        # Differential Learning Rates / Fine-Tuning
        for param in self.backbone.parameters():
            param.requires_grad = False

        for name, child in self.backbone.named_children():
            if name == "blocks":
                for i, block in enumerate(child):
                    if i >= 5:
                        for param in block.parameters():
                            param.requires_grad = True
            elif name in ["conv_head", "bn2"]:
                for param in child.parameters():
                    param.requires_grad = True

        # --- Stream A: Deep Interaction Stream (Non-Linear Correction) ---
        # Inputs: Image Embedding + All Tabular Features
        # Tabular: [Base_FVC, Base_Pct, Rel_Time, Age, Sex, Smoke] (6 features)
        self.deep_input_dim = Config.IMG_EMBED_DIM + 6
        self.deep_mlp = nn.Sequential(
            nn.Linear(self.deep_input_dim, Config.VISUAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.VISUAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # --- Stream B: Linear Residual Stream (Autoregressive Trend) ---
        # Inputs: Only Baseline FVC (idx 0) and Relative Weeks (idx 2)
        # Cite solution_lesson_node_00060: Over-Parameterization of Linear Baselines
        # We project the 2 linear features into the latent dimension instead of direct output
        self.linear_input_dim = 2
        self.linear_project = nn.Linear(self.linear_input_dim, Config.LATENT_DIM)
        # We initialize this projection to be close to identity/zeros to let gradient flow easily
        # but standard init is usually fine with residual sum.

        # --- Shared Head ---
        # Projects fused latent to Mu and Sigma
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, tabular):
        """
        Args:
            image: Tensor (Batch, 3, H, W)
            tabular: Tensor (Batch, 6)
        """
        # 1. Image Features
        x = self.backbone.forward_features(image)
        x = self.global_pool(x).flatten(1)
        img_embed = self.img_project(x)

        # 2. Stream A: Deep Interaction
        # Concatenate image and all tabular data
        deep_in = torch.cat([img_embed, tabular], dim=1)
        h_deep = self.deep_mlp(deep_in)

        # 3. Stream B: Linear Residual
        # Extract Baseline FVC (0) and Relative Weeks (2)
        # tabular is [Base_FVC, Base_Pct, Rel_Time, Age, Sex, Smoke]
        linear_feats = tabular[:, [0, 2]]
        h_linear = self.linear_project(linear_feats)

        # 4. Fusion (Summation)
        # Prediction = LinearTrend + DeepCorrection
        h_final = h_deep + h_linear

        # 5. Output
        out = self.head(h_final)

        mu = out[:, 0].unsqueeze(1)
        raw_sigma = out[:, 1].unsqueeze(1)

        # Softplus for positivity
        sigma = F.softplus(raw_sigma) + 1e-6

        return torch.cat([mu, sigma], dim=1)
