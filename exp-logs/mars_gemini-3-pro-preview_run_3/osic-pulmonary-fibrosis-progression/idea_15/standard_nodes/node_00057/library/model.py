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
    Explicit Additive Dual-Stream Network.
    Sums outputs from a robust Linear Wide Stream and a complex Deep Stream.
    """

    def __init__(self):
        super().__init__()

        self.image_encoder = ImageEncoder()

        # --- Stream A: Wide Linear Branch ---
        # Inputs: [Baseline_FVC, Relative_Time] -> 2 features
        # Outputs: [mu_wide, sigma_logit_wide] -> 2 outputs
        self.wide_stream = nn.Linear(2, 2)

        # --- Stream B: Deep Interaction Branch ---
        # Inputs:
        #   Image Projection (128)
        #   Tabular Features (4) -> [Baseline_FVC, Age, Sex, Smoking]
        #   Relative Time (1)
        # Total: 128 + 4 + 1 = 133
        deep_input_dim = Config.PROJECTION_DIM + 4 + 1

        self.deep_stream = nn.Sequential(
            nn.Linear(deep_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 2),  # [mu_deep, sigma_logit_deep]
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initializes the output layers to ensure stable training start.
        Crucially, we initialize the sigma bias to a positive value to start
        with high uncertainty, preventing 1/sigma explosion in NLL loss.
        """
        # 1. Wide Stream Initialization
        # Zero weights so it starts as a neutral pass-through (predicting 0 mean)
        nn.init.zeros_(self.wide_stream.weight)
        nn.init.zeros_(self.wide_stream.bias)
        # Set sigma bias to +3.0 -> softplus(3.0) approx 3.0
        # This corresponds to ~3 standard deviations of uncertainty, which is safe.
        self.wide_stream.bias.data[1] = 3.0

        # 2. Deep Stream Initialization
        # Access the last linear layer
        last_layer = self.deep_stream[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        # Set sigma bias to +3.0
        last_layer.bias.data[1] = 3.0

    def forward(self, image, tabular, time):
        """
        Args:
            image: (Batch, 3, H, W)
            tabular: (Batch, 4) -> [Baseline_FVC, Age, Sex, SmokingStatus]
            time: (Batch, 1) -> [Relative_Weeks]

        Returns:
            mu_final: Predicted FVC (scaled)
            sigma_final: Predicted Confidence (scaled)
        """
        # 1. Extract Image Features
        img_embed = self.image_encoder(image)  # (Batch, 128)

        # 2. Prepare Wide Stream Inputs
        # Select Baseline_FVC (index 0) and Time
        baseline_fvc = tabular[:, 0:1]
        wide_inputs = torch.cat([baseline_fvc, time], dim=1)  # (Batch, 2)

        # 3. Prepare Deep Stream Inputs
        deep_inputs = torch.cat([img_embed, tabular, time], dim=1)  # (Batch, 133)

        # 4. Compute Stream Outputs
        # shape: (Batch, 2)
        wide_out = self.wide_stream(wide_inputs)
        deep_out = self.deep_stream(deep_inputs)

        # Split into Mean and Sigma-Logit
        mu_wide, sigma_logit_wide = wide_out[:, 0:1], wide_out[:, 1:2]
        mu_deep, sigma_logit_deep = deep_out[:, 0:1], deep_out[:, 1:2]

        # 5. Explicit Additive Fusion
        # Mean: Direct Sum
        mu_final = mu_wide + mu_deep

        # Uncertainty: Sum logits, then Softplus + Epsilon
        # This allows the deep branch to increase OR decrease uncertainty
        # relative to the linear time-dependent trend.
        sigma_logit_sum = sigma_logit_wide + sigma_logit_deep
        sigma_final = F.softplus(sigma_logit_sum) + Config.EPSILON

        return mu_final, sigma_final
