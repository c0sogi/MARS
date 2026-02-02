import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer (2D).
    Computes the p-norm of the input tensor.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Pools over H and W
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class SETIModel(nn.Module):
    """
    EfficientNet-V2 Small with GeM Pooling for SETI signal detection.
    """

    def __init__(self, pretrained=True):
        super(SETIModel, self).__init__()

        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=pretrained,
            in_chans=Config.in_channels,
            num_classes=0,
            global_pool="",
        )

        in_features = self.backbone.num_features

        # Generalized Mean Pooling
        self.gem = GeM(p=3.0)

        # Classification Head
        self.drop = nn.Dropout(Config.drop_rate)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # x: (B, 3, 1638, 256)
        x = self.backbone(x)  # (B, C, H', W')
        x = self.gem(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        logits = self.fc(x)

        return logits
