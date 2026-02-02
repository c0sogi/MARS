import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class PRTNet(nn.Module):
    """
    Point-Wise Relative-Time Fusion Network (PRT-Net).

    A hybrid CNN-MLP architecture that predicts FVC mean and confidence based on:
    1. A content-adaptive 2.5D CT scan selection (processed by EfficientNet-B0).
    2. Static clinical metadata (Baseline FVC, Age, Sex, Smoking).
    3. Relative time from baseline.

    The architecture explicitly avoids normalization layers in the head to preserve
    the magnitude of the baseline FVC signal.
    """

    def __init__(self):
        super(PRTNet, self).__init__()

        # ---------------------------------------------------------------------
        # 1. Image Backbone (EfficientNet-B0)
        # ---------------------------------------------------------------------
        # num_classes=0 ensures the model returns the pooled feature vector (1280 dim)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=3,
        )

        # Feature dimension for EfficientNet-B0 is 1280
        self.backbone_dim = self.backbone.num_features

        # ---------------------------------------------------------------------
        # 2. Differential Freezing Strategy
        # ---------------------------------------------------------------------
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the top layers for domain adaptation
        # Unfreeze the head convolution and batch norm
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        # Unfreeze the last few blocks (e.g., the last 2 blocks of the 7 blocks in B0)
        # EfficientNet blocks are stored in a Sequential container named 'blocks'
        # We unfreeze the last 2 blocks to allow texture learning
        for block in self.backbone.blocks[-2:]:
            for param in block.parameters():
                param.requires_grad = True

        # ---------------------------------------------------------------------
        # 3. Fusion Components
        # ---------------------------------------------------------------------
        # Project high-dim image features to lower dim to balance with tabular data
        self.img_projector = nn.Linear(self.backbone_dim, Config.IMG_PROJ_DIM)

        # Calculate total input dimension for the MLP
        # Image Projection (128) + Static Features (4) + Relative Time (1)
        # Static features: Baseline_FVC, Age, Sex, Smoking
        self.n_static = 4
        self.n_time = 1
        self.fusion_dim = Config.IMG_PROJ_DIM + self.n_static + self.n_time

        # ---------------------------------------------------------------------
        # 4. MLP Head (Point-Wise Predictor)
        # ---------------------------------------------------------------------
        # Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> Output
        # NO Batch/Layer Norm to preserve magnitude information

        layers = []
        in_dim = self.fusion_dim

        for hidden_dim in Config.FUSION_HIDDEN_DIMS:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        # Final projection to 2 values: mu (mean) and raw_sigma (confidence)
        layers.append(nn.Linear(in_dim, 2))

        self.mlp = nn.Sequential(*layers)

    def forward(self, image, static, rel_time):
        """
        Forward pass.

        Args:
            image (torch.Tensor): (B, 3, H, W) - 2.5D CT slices
            static (torch.Tensor): (B, 4) - [Baseline_FVC, Age, Sex, Smoking]
            rel_time (torch.Tensor): (B, 1) - Scaled relative weeks

        Returns:
            mu (torch.Tensor): Predicted standardized FVC (B,)
            sigma (torch.Tensor): Predicted standardized confidence (B,)
        """
        # 1. Image Feature Extraction
        # Backbone returns global pooled features (B, 1280)
        img_feats = self.backbone(image)

        # Project to lower dimension (B, 128)
        img_emb = self.img_projector(img_feats)

        # 2. Feature Fusion
        # Concatenate: [Image(128), Static(4), Time(1)]
        # Ensure dimensions match for concatenation
        combined = torch.cat([img_emb, static, rel_time], dim=1)

        # 3. MLP Prediction
        out = self.mlp(combined)

        # 4. Output Parsing
        # out[:, 0] -> mu (Mean FVC)
        # out[:, 1] -> raw_sigma (Confidence)
        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Enforce positive sigma using Softplus + Offset
        # We do not clip to 70 here; that is done in the metric/post-processing.
        # This allows gradients to flow naturally.
        sigma = F.softplus(raw_sigma) + Config.SIGMA_OFFSET

        return mu, sigma
