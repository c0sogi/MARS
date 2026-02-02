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


class WhaleClassifier(nn.Module):
    """
    Single-Stream architecture using EfficientNet backbone with GeM Pooling.
    """

    def __init__(self, config=Config):
        super(WhaleClassifier, self).__init__()

        # Backbone
        self.backbone = timm.create_model(
            config.BACKBONE, pretrained=config.PRETRAINED, num_classes=0, global_pool=""
        )

        # Input adapter: 1 channel (spectrogram) -> 3 channels (RGB expected by EffNet)
        self.adapter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # Pooling Layer
        self.pool = GeM(p=config.GEM_P)

        # Feature dimension check
        dummy_in = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feat_dim = self.backbone(dummy_in).shape[1]

        # Classification Head
        self.fc = nn.Sequential(nn.Dropout(config.DROPOUT), nn.Linear(feat_dim, 1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input Spectrogram (B, 1, F, T)
        """
        x = self.adapter(x)
        x = self.backbone(x)  # Output: (B, C, H, W)
        x = self.pool(x).flatten(1)  # Output: (B, C)
        output = self.fc(x)  # Output: (B, 1)

        return output
