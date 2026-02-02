import torch
import torch.nn as nn
import timm
from library.config import Config


class MGSHDNetwork(nn.Module):
    """
    Modality-Grouped Stabilized High-Density (MG-SHD) Network.

    Architecture:
    1. Input: (B, 128, 224, 224) - 32 slices * 4 modalities.
    2. Stem: Stabilized Global-Mixing Stem (Conv 128->64, BN, ReLU).
    3. Backbone: EfficientNet-B0 (in_chans=64, pretrained).
    4. Head: Global Avg Pool -> Linear -> Logits.
    """

    def __init__(self):
        super().__init__()

        # ==========================================
        # 1. Stabilized Global-Mixing Stem
        # ==========================================
        # Reduces high-density input (128 channels) to a standard feature depth (64)
        # while performing early global mixing across all slices and modalities.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=Config.IN_CHANNELS,  # 128
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Explicit Kaiming/He Normal Initialization for stability
        nn.init.kaiming_normal_(
            self.stem[0].weight, mode="fan_out", nonlinearity="relu"
        )

        # ==========================================
        # 2. Backbone (EfficientNet-B0)
        # ==========================================
        # Configured to accept 64 channels from the stem.
        # drop_path_rate enabled for regularization.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=True,
            in_chans=64,
            drop_path_rate=Config.DROP_PATH_RATE,
            num_classes=0,  # Remove default classifier
            global_pool="",  # Return feature maps, we handle pooling
        )

        # Determine the number of output features from the backbone
        # EfficientNet-B0 typically outputs 1280 channels
        with torch.no_grad():
            dummy_input = torch.randn(1, 64, 112, 112)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # ==========================================
        # 3. Head
        # ==========================================
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.num_features, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 224, 224).
        Returns:
            torch.Tensor: Logits of shape (B, 1).
        """
        # 1. Pass through Stabilized Stem
        # Shape: (B, 128, 224, 224) -> (B, 64, 112, 112)
        x = self.stem(x)

        # 2. Pass through Backbone
        # Shape: (B, 64, 112, 112) -> (B, 1280, 7, 7)
        x = self.backbone(x)

        # 3. Head
        # Shape: (B, 1280, 7, 7) -> (B, 1280, 1, 1)
        x = self.global_pool(x)

        # Flatten: (B, 1280)
        x = x.flatten(1)

        # Linear Projection: (B, 1)
        logits = self.fc(x)

        return logits
