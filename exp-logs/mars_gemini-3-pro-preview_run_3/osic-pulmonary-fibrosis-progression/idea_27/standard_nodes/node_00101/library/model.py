import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class GMARNet(nn.Module):
    """
    Gated Metric-Aligned Residual Network (GMAR-Net).

    A hybrid CNN-MLP architecture that uses a clinical controller stream to
    gate visual features extracted from CT scans.
    """

    def __init__(self):
        super(GMARNet, self).__init__()

        # ---------------------------------------------------------------------
        # Image Branch (Fine-Tuned Content-Adaptive 2.5D)
        # ---------------------------------------------------------------------
        # Backbone: EfficientNet-B2
        # Input: (Batch, 3, 260, 260) -> treated as 3-channel 2D image
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top two stages of the feature extractor
        # efficientnet_b2.features is a Sequential container.
        # We unfreeze the last few blocks to allow high-level feature adaptation.
        # Indices 7 and 8 correspond to the final MBConv stage and the Conv1x1 expansion.
        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True

        # Replace classifier with projection head
        # EfficientNet classifier is usually Dropout -> Linear.
        # We want GlobalAvgPool (handled by backbone.avgpool) -> Flatten -> Linear
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.backbone.classifier[1].in_features, Config.IMAGE_PROJ_DIM),
        )

        # ---------------------------------------------------------------------
        # Stream A: Over-Parameterized Clinical Anchor (The Controller)
        # ---------------------------------------------------------------------
        # Input: Baseline FVC, Relative Time, Age, Sex, SmokingStatus (5 features)
        # Note: Config.CLINICAL_INPUT_DIM is 4, but Data loader provides 5.
        # We use 5 to match the actual data.
        self.clinical_input_dim = 5

        self.clinical_net = nn.Sequential(
            nn.Linear(self.clinical_input_dim, Config.CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.CLINICAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # Linear Residual Stream for Clinical Data (Cite solution_lesson_node_00052, solution_lesson_node_00060)
        # Preserves strong linear signal from Baseline FVC and Time without ReLU blocking
        self.clinical_residual = nn.Linear(self.clinical_input_dim, Config.LATENT_DIM)

        # Gating Mechanism: Projects H_clin to a gating vector G
        self.gate_proj = nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM)

        # ---------------------------------------------------------------------
        # Stream B: Cascaded Visual Interaction
        # ---------------------------------------------------------------------
        # Input: Concatenation of [Image Projection, H_clin]
        visual_input_dim = Config.IMAGE_PROJ_DIM + Config.LATENT_DIM

        self.visual_net = nn.Sequential(
            nn.Linear(visual_input_dim, Config.LATENT_DIM),
            nn.ReLU(),
            nn.Linear(Config.LATENT_DIM, Config.LATENT_DIM),
        )

        # ---------------------------------------------------------------------
        # Metric-Aligned Head
        # ---------------------------------------------------------------------
        # Projects H_final to mu and sigma
        # No time-shortcut is used here.
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, clinical):
        """
        Args:
            image (torch.Tensor): (Batch, 3, H, W)
            clinical (torch.Tensor): (Batch, 5)

        Returns:
            torch.Tensor: (Batch, 2) -> [FVC_pred, Sigma_pred]
        """
        # 1. Image Branch
        # Extract features
        img_feats = self.backbone(image)  # (Batch, IMAGE_PROJ_DIM)

        # 2. Stream A (Clinical)
        h_clin_mlp = self.clinical_net(clinical)  # (Batch, LATENT_DIM)
        h_clin_res = self.clinical_residual(clinical)  # (Batch, LATENT_DIM)

        # Compute Gate
        # G = Sigmoid(Linear(H_clin))
        gate = torch.sigmoid(self.gate_proj(h_clin_mlp))  # (Batch, LATENT_DIM)

        # 3. Stream B (Visual)
        # Concatenate Image features and Clinical Latent
        vis_input = torch.cat(
            [img_feats, h_clin_mlp], dim=1
        )  # (Batch, IMAGE_PROJ_DIM + LATENT_DIM)
        h_vis = self.visual_net(vis_input)  # (Batch, LATENT_DIM)

        # 4. Gated Fusion
        # H_final = H_clin_res + (G * H_vis)
        # We use the Linear Residual as the base to allow negative values (standardized FVC) to pass through.
        h_final = h_clin_res + (gate * h_vis)

        # 5. Prediction Head
        out = self.head(h_final)

        # Split into mu and raw sigma
        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Enforce positivity for sigma
        # We use softplus + epsilon to avoid numerical instability
        sigma = F.softplus(raw_sigma) + 1e-6

        # Stack back to (Batch, 2)
        return torch.stack([mu, sigma], dim=1)
