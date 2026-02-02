import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class ImageEncoder(nn.Module):
    """
    Extracts features from CT scans using a fine-tuned EfficientNet-B2.
    Top layers are unfrozen to allow domain adaptation.
    """

    def __init__(self):
        super().__init__()
        # Load backbone with num_classes=0 to get pooled features
        # in_chans=3 matches our input of 3 stacked slices
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            in_chans=Config.NUM_SLICES,
        )

        # EfficientNet-B2 outputs 1408 features after global pooling
        self.in_features = self.backbone.num_features

        # Projection head to reduce dimensionality before fusion
        self.projection = nn.Linear(self.in_features, Config.PROJECTION_DIM)

        self._set_trainable_layers()

    def _set_trainable_layers(self):
        """
        Freezes the entire backbone, then unfreezes the top two stages
        and the head to preserve low-level edge detection features.
        """
        # 1. Freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the Head (Conv + BN)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # 3. Unfreeze the last 2 stages of blocks
        # timm's EfficientNet stores blocks in a Sequential container.
        # We iterate and unfreeze the last 2 blocks.
        num_blocks = len(self.backbone.blocks)
        # Ensure we don't go out of bounds if model is small (unlikely for B2)
        start_idx = max(0, num_blocks - 2)

        for i in range(start_idx, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        features = self.backbone(x)  # (Batch, 1408)
        projected = self.projection(features)  # (Batch, 128)
        return projected


class EADSNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network (DSPRNet).
    Implements a Linear Residual Stream for mean prediction and a Shared Input Head for uncertainty.
    """

    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoder()

        # --- Deep Stream Feature Extractor ---
        # Inputs: Img(128) + Tabular(4) + Time(1) = 133
        deep_input_dim = Config.PROJECTION_DIM + 4 + 1
        self.deep_stream = nn.Sequential(
            nn.Linear(deep_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # --- Mu Heads ---
        # Linear Residual: [Baseline_FVC, Time] -> Mu
        # Cite solution_lesson_node_00052: Linear residual for strong autoregressive signals
        self.mu_linear = nn.Linear(2, 1)

        # Deep Correction: [Deep_Feat] -> Mu
        self.mu_deep = nn.Linear(128, 1)

        # --- Sigma Head ---
        # Shared Input: [Deep_Feat(128), Baseline_FVC(1), Time(1)] -> 130
        # Cite solution_lesson_node_00055: Direct pathway for strong priors to uncertainty
        self.sigma_head = nn.Sequential(nn.Linear(130, 64), nn.ReLU(), nn.Linear(64, 1))

        self._init_weights()

    def _init_weights(self):
        """
        Initializes weights to start with a robust baseline.
        """
        # 1. Mu Linear: Initialize to Identity for Baseline_FVC
        # We want mu approx Baseline_FVC at start.
        nn.init.zeros_(self.mu_linear.weight)
        nn.init.zeros_(self.mu_linear.bias)
        with torch.no_grad():
            self.mu_linear.weight[0, 0] = 1.0  # Weight for Baseline_FVC

        # 2. Mu Deep: Initialize to Zero (start with no correction)
        nn.init.zeros_(self.mu_deep.weight)
        nn.init.zeros_(self.mu_deep.bias)

        # 3. Sigma Head: Initialize to high uncertainty
        # Cite solution_lesson_node_00001: Bias Initialization
        for m in self.sigma_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Set final bias to +3.0 (softplus(3) approx 3.0)
        self.sigma_head[-1].bias.data.fill_(3.0)

    def forward(self, image, tabular, time):
        # 1. Extract Image Features
        img_feat = self.image_encoder(image)  # (Batch, 128)

        # 2. Deep Stream Features
        deep_in = torch.cat([img_feat, tabular, time], dim=1)
        deep_feat = self.deep_stream(deep_in)  # (Batch, 128)

        # 3. Linear Inputs (Baseline, Time)
        # tabular is [Baseline, Age, Sex, Smoke]
        linear_in = torch.cat([tabular[:, 0:1], time], dim=1)

        # 4. Mu Calculation
        # Prediction = LinearTrend(Base, Time) + DeepCorrection(Image, Context)
        mu = self.mu_linear(linear_in) + self.mu_deep(deep_feat)

        # 5. Sigma Calculation
        # Concatenate Deep Features with Strong Priors (Base, Time)
        sigma_in = torch.cat([deep_feat, linear_in], dim=1)
        sigma_logit = self.sigma_head(sigma_in)
        sigma = F.softplus(sigma_logit) + Config.EPSILON

        return mu, sigma
