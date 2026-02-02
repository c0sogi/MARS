import torch
import torch.nn as nn
import timm
from library.config import Config


class ClinicalStream(nn.Module):
    """
    Stream A: Over-Parameterized Clinical Anchor (The Base Learner).
    Learns the baseline disease trajectory based on clinical scalars.
    Input: 5 Clinical Features
    Output: Base Mean, Base Raw Sigma
    """

    def __init__(self, input_dim=5):
        super(ClinicalStream, self).__init__()
        # Architecture: Linear(Input -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> 2)
        # We include ReLU after the 128->64 layer to ensure it functions as a proper MLP hidden layer.
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class VisualStream(nn.Module):
    """
    Stream B: Regularized Visual Residual (The Correction).
    Uses EfficientNet-B2 to learn residual corrections to the clinical baseline.
    Input: Image (Batch, 3, H, W), Tabular (Batch, 5)
    Output: Residual Mean, Residual Raw Sigma
    """

    def __init__(self, tabular_dim=5):
        super(VisualStream, self).__init__()

        # Backbone: EfficientNet-B2
        # num_classes=0 ensures we get the Global Average Pooled features (1408 dim)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, in_chans=3, num_classes=0
        )

        # Feature dimension for EfficientNet-B2
        self.backbone_dim = self.backbone.num_features

        # Unfreezing Logic: Freeze all, then unfreeze top two convolutional stages
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last two blocks (stages) in the EfficientNet hierarchy
        # timm stores blocks in a Sequential named 'blocks'
        if hasattr(self.backbone, "blocks"):
            # Unfreeze last 2 stages
            for stage in self.backbone.blocks[-2:]:
                for param in stage.parameters():
                    param.requires_grad = True

        # Unfreeze the conv_head and bn2 (top-most layers before pooling)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Bottleneck Projection: 1408 -> 64
        self.projection = nn.Linear(self.backbone_dim, 64)

        # MLP Head with Context Injection
        # Input: Projected Image (64) + Clinical Scalars (5)
        fused_dim = 64 + tabular_dim

        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2),
        )

        # Zero-Initialization: Initialize final layer weights/bias to 0
        # This ensures the residual stream starts with 0 output (Boosting paradigm)
        final_layer = self.mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, img, tabular):
        # 1. Extract Visual Features
        features = self.backbone(img)  # (Batch, 1408)

        # 2. Project to lower dimension
        proj = self.projection(features)  # (Batch, 64)

        # 3. Context Injection: Concatenate with clinical scalars
        fused = torch.cat([proj, tabular], dim=1)  # (Batch, 69)

        # 4. Predict Residuals
        out = self.mlp(fused)  # (Batch, 2)
        return out


class RODSNet(nn.Module):
    """
    Regularized Output-Space Dual-Stream Network.
    Fuses Clinical Anchor and Visual Residual via summation.
    """

    def __init__(self):
        super(RODSNet, self).__init__()
        self.clinical_stream = ClinicalStream(input_dim=5)
        self.visual_stream = VisualStream(tabular_dim=5)

    def forward(self, img, tabular):
        """
        Args:
            img (torch.Tensor): CT Image slices (Batch, 3, H, W)
            tabular (torch.Tensor): Clinical features (Batch, 5)

        Returns:
            torch.Tensor: Combined output (Batch, 2) -> [Mean, Raw_Sigma]
        """
        # Stream A: Clinical Anchor (Base Prediction)
        base_out = self.clinical_stream(tabular)

        # Stream B: Visual Residual (Correction)
        residual_out = self.visual_stream(img, tabular)

        # Output Fusion: Summation in the raw output space
        # Mean_final = Mean_base + Mean_residual
        # Raw_Sigma_final = Raw_Sigma_base + Raw_Sigma_residual
        # The loss function handles the Softplus conversion for Sigma.
        return base_out + residual_out
