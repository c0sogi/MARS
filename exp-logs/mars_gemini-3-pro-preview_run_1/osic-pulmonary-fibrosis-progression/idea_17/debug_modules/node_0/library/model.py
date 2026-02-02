import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class VisualBackbone(nn.Module):
    """
    Extracts high-fidelity global features from CT slices using EfficientNet-B0.
    Maintains the native 1280-dim output without bottleneck projection.
    """

    def __init__(self):
        super(VisualBackbone, self).__init__()
        # Load EfficientNet-B0 pre-trained on ImageNet
        # num_classes=0 ensures we get the Global Average Pooled feature vector
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, in_chans=Config.IN_CHANNELS
        )

    def forward(self, x):
        # x shape: (Batch, 3, 224, 224)
        # Output shape: (Batch, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Projects low-dimensional clinical metadata UP to the high-dimensional visual space.
    """

    def __init__(self, input_dim=7, output_dim=Config.FEATURE_DIM):
        super(TabularEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x shape: (Batch, 7)
        # Output shape: (Batch, 1280)
        return self.net(x)


class DualAxisNet(nn.Module):
    """
    Full-Fidelity Concatenated Dual-Axis Network.

    Architecture:
    1. Independent Visual Backbones for Axial and Coronal views.
    2. Up-projection of Tabular data.
    3. Transformer Fusion (Sequence Length 3).
    4. Concatenated Readout (Flattening).
    5. Skip Connection for raw priors.
    6. Parametric Regression Head.
    """

    def __init__(self):
        super(DualAxisNet, self).__init__()

        # 1. Independent Visual Backbones
        self.axial_backbone = VisualBackbone()
        self.coronal_backbone = VisualBackbone()

        # 2. Tabular Encoder
        # Input dim 7 comes from dataset.py (Age, Pct, Sex(2), Smoke(3))
        self.tabular_encoder = TabularEncoder(
            input_dim=7, output_dim=Config.FEATURE_DIM
        )

        # 3. Transformer Fusion
        # d_model = 1280 (Native B0 dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.FEATURE_DIM,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.fusion_layer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 4. Regression Head
        # Input: Flattened Transformer Output (3 * 1280) + Skip Connection (2)
        # 3 * 1280 = 3840
        # Skip features: Baseline_FVC_Scaled, Percent_Scaled
        head_input_dim = (3 * Config.FEATURE_DIM) + 2

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # Alpha, Sigma_Base, Sigma_Growth
        )

    def forward(self, batch):
        """
        Args:
            batch (dict): Dictionary containing:
                - 'axial': (B, 3, 224, 224)
                - 'coronal': (B, 3, 224, 224)
                - 'tabular': (B, 7)
                - 'skip': (B, 2)

        Returns:
            torch.Tensor: (B, 3) containing [alpha, sigma_base, sigma_growth]
        """
        axial_img = batch["axial"]
        coronal_img = batch["coronal"]
        tabular_feats = batch["tabular"]
        skip_feats = batch["skip"]

        # 1. Feature Extraction
        # (B, 1280)
        v_ax = self.axial_backbone(axial_img)
        # (B, 1280)
        v_cor = self.coronal_backbone(coronal_img)
        # (B, 1280)
        v_tab = self.tabular_encoder(tabular_feats)

        # 2. Sequence Construction
        # Stack to form sequence: [Axial, Coronal, Tabular]
        # Shape: (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # 3. Fusion (Symmetric Attention)
        # Shape: (B, 3, 1280)
        context_seq = self.fusion_layer(seq)

        # 4. Concatenated Readout
        # Flatten: (B, 3840)
        flat_context = context_seq.view(context_seq.size(0), -1)

        # 5. Skip Connection
        # Concatenate with raw priors: (B, 3842)
        combined = torch.cat([flat_context, skip_feats], dim=1)

        # 6. Prediction
        out = self.head(combined)

        # 7. Constrained Output
        # out[:, 0] is Alpha (Slope) -> Linear (can be negative or positive)
        # out[:, 1] is Sigma Base -> Softplus (Must be positive)
        # out[:, 2] is Sigma Growth -> Softplus (Must be positive)

        alpha = out[:, 0].unsqueeze(1)
        sigma_base = F.softplus(out[:, 1]).unsqueeze(1)
        sigma_growth = F.softplus(out[:, 2]).unsqueeze(1)

        return torch.cat([alpha, sigma_base, sigma_growth], dim=1)
