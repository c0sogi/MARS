import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean over the spatial dimensions of the input.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Pool over H, W -> (B, C, 1, 1)
        # We use the spatial dimensions of x for the pooling kernel size
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class DualStreamEfficientNet(nn.Module):
    """
    Dual-Stream architecture processing spectral and temporal features independently
    before fusing them for classification.
    """

    def __init__(self, config=Config):
        super(DualStreamEfficientNet, self).__init__()

        # Stream 1: Spectral (High Frequency Resolution)
        # We remove the global pooling and classifier to get feature maps
        self.backbone1 = timm.create_model(
            config.BACKBONE, pretrained=config.PRETRAINED, num_classes=0, global_pool=""
        )
        # Input adapter: 1 channel (spectrogram) -> 3 channels (RGB expected by EffNet)
        self.adapter1 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # Stream 2: Temporal (High Temporal Resolution)
        self.backbone2 = timm.create_model(
            config.BACKBONE, pretrained=config.PRETRAINED, num_classes=0, global_pool=""
        )
        # Input adapter: 1 channel -> 3 channels
        self.adapter2 = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # Pooling Layer
        self.pool = GeM(p=config.GEM_P)

        # Feature dimension check to dynamically set Linear layer input size
        # We assume both backbones are identical in architecture
        dummy_in = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            # Pass through one backbone to get feature map channels (e.g., 1280 for EffNetV2-M)
            feat_dim = self.backbone1(dummy_in).shape[1]

        # Classification Head
        # Concatenates features from both streams, so input dim is feat_dim * 2
        self.fc = nn.Sequential(nn.Dropout(config.DROPOUT), nn.Linear(feat_dim * 2, 1))

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): Input for Spectral Stream (B, 1, F, T)
            x2 (torch.Tensor): Input for Temporal Stream (B, 1, F, T)
        """
        # --- Stream 1: Spectral ---
        x1 = self.adapter1(x1)
        f1 = self.backbone1(x1)  # Output: (B, C, H, W)
        f1 = self.pool(f1).flatten(1)  # Output: (B, C)

        # --- Stream 2: Temporal ---
        x2 = self.adapter2(x2)
        f2 = self.backbone2(x2)  # Output: (B, C, H, W)
        f2 = self.pool(f2).flatten(1)  # Output: (B, C)

        # --- Fusion & Classification ---
        concat = torch.cat([f1, f2], dim=1)  # Output: (B, 2*C)
        output = self.fc(concat)  # Output: (B, 1)

        return output
