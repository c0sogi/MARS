import torch
import torch.nn as nn
import timm
from library.config import Config


class RMSHDNet(nn.Module):
    """
    Robust Modality-Structured High-Density (RMS-HD) Network.

    This architecture is designed to process high-density volumetric MRI data (128 channels)
    by first compressing it into a stable feature space using a specialized stem,
    and then extracting features using an EfficientNet backbone.
    """

    def __init__(self):
        super(RMSHDNet, self).__init__()

        # ==========================================
        # 1. Stabilized Global-Mixing Adapter (Stem)
        # ==========================================
        # Function:
        #   - Compresses 128 input channels to 64 (Stability Sweet Spot)
        #   - Mixes information from all slices/modalities via 3x3 Conv
        #   - Downsamples resolution 224 -> 112
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=Config.IN_CHANNELS,
                out_channels=Config.STEM_OUT_CHANNELS,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(Config.STEM_OUT_CHANNELS),
            nn.ReLU(inplace=True),
        )

        # Explicit initialization is critical for high-channel inputs
        self._init_stem()

        # ==========================================
        # 2. Backbone (EfficientNet-B0)
        # ==========================================
        # Configured to accept the 64-channel output from the stem.
        # global_pool='avg' ensures we get a feature vector (B, num_features).
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.PRETRAINED,
            in_chans=Config.STEM_OUT_CHANNELS,
            drop_path_rate=Config.DROP_PATH_RATE,
            num_classes=0,  # Return features (pool applied via global_pool)
            global_pool="avg",
        )

        # ==========================================
        # 3. Classification Head
        # ==========================================
        # Projects features to a single logit.
        # Note: Sigmoid is not applied here to allow use of BCEWithLogitsLoss.
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def _init_stem(self):
        """
        Applies Kaiming/He Normal initialization to the stem convolution.
        This prevents gradient explosion/vanishing when projecting the
        high-dimensional (128ch) input into the lower-dimensional (64ch) space.
        """
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 224, 224)

        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # 1. Stem: (B, 128, 224, 224) -> (B, 64, 112, 112)
        x = self.stem(x)

        # 2. Backbone: (B, 64, 112, 112) -> (B, num_features)
        x = self.backbone(x)

        # 3. Head: (B, num_features) -> (B, 1)
        x = self.fc(x)

        return x
