import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class GMARNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    (Class name kept as GMARNet for compatibility with existing scripts).

    Uses two parallel streams:
    1. Deep Stream: Fuses Image and All Clinical Features for complex interactions.
    2. Linear Stream: Projects strong autoregressive priors (Baseline FVC, Time)
       into the latent space (Cite solution_lesson_node_00052, solution_lesson_node_00060).
    """

    def __init__(self):
        super(GMARNet, self).__init__()

        # ---------------------------------------------------------------------
        # Image Branch (Fine-Tuned Content-Adaptive 2.5D)
        # ---------------------------------------------------------------------
        # Backbone: EfficientNet-B2 (Cite solution_lesson_node_00071)
        weights = models.EfficientNet_B2_Weights.DEFAULT
        self.backbone = models.efficientnet_b2(weights=weights)

        # Freeze all layers first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top two stages (Cite solution_lesson_node_00027)
        for param in self.backbone.features[-2:].parameters():
            param.requires_grad = True

        # Projection Head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(self.backbone.classifier[1].in_features, Config.IMAGE_PROJ_DIM),
        )

        # ---------------------------------------------------------------------
        # Stream 1: Deep Interaction Stream
        # ---------------------------------------------------------------------
        # Input: Concatenation of [Image Embed (64), Clinical Raw (5)]
        # We use direct concatenation of raw scalars (Cite solution_lesson_node_00035)
        deep_input_dim = Config.IMAGE_PROJ_DIM + Config.CLINICAL_INPUT_DIM

        self.deep_stream = nn.Sequential(
            nn.Linear(deep_input_dim, Config.CLINICAL_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.CLINICAL_HIDDEN_DIM, Config.LATENT_DIM),
        )

        # ---------------------------------------------------------------------
        # Stream 2: Over-Parameterized Linear Residual Stream
        # ---------------------------------------------------------------------
        # Input: Baseline FVC, Relative Time (2 features)
        # Projects to Latent Dim to avoid bottlenecks (Cite solution_lesson_node_00060)
        self.linear_stream = nn.Linear(Config.LINEAR_INPUT_DIM, Config.LATENT_DIM)

        # ---------------------------------------------------------------------
        # Shared Head
        # ---------------------------------------------------------------------
        # Predicts both mu and sigma from the fused representation (Cite solution_lesson_node_00055)
        self.head = nn.Linear(Config.LATENT_DIM, 2)

    def forward(self, image, clinical):
        """
        Args:
            image (torch.Tensor): (Batch, 3, H, W)
            clinical (torch.Tensor): (Batch, 5)
                                     [Baseline_FVC, Time, Age, Sex, Smoking]
        """
        # 1. Image Features
        img_feats = self.backbone(image)  # (Batch, 64)

        # 2. Deep Stream (Complex Interactions)
        # Concatenate Image + All Clinical
        deep_input = torch.cat([img_feats, clinical], dim=1)
        h_deep = self.deep_stream(deep_input)  # (Batch, 64)

        # 3. Linear Stream (Autoregressive Residuals)
        # Extract Baseline FVC (idx 0) and Time (idx 1)
        linear_input = clinical[:, :2]
        h_linear = self.linear_stream(linear_input)  # (Batch, 64)

        # 4. Residual Fusion (Cite solution_lesson_node_00052)
        h_final = h_deep + h_linear

        # 5. Prediction Head
        out = self.head(h_final)

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Enforce positivity (Cite solution_lesson_node_00013)
        # Using softplus + epsilon. Clipping at 70 is done in metric/post-processing.
        sigma = F.softplus(raw_sigma) + 1e-6

        return torch.stack([mu, sigma], dim=1)
