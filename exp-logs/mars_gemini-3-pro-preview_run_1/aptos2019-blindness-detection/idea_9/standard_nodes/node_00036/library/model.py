import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Clamp input to avoid numerical instability with pow
        # Global Average Pooling of x^p, then raised to 1/p
        # x shape: [B, C, H, W]
        # avg_pool2d output: [B, C, 1, 1]
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        x = x.pow(1.0 / self.p)
        return x


class RetinaModel(nn.Module):
    """
    RetinaModel for Diabetic Retinopathy Severity Prediction.

    Architecture:
    1. Backbone: ConvNeXt-Base (timm)
    2. Pooling: Dual-Stream (Concatenation of GAP and GeM)
    3. Head: LayerNorm -> Multi-Sample Dropout -> Linear (Ordinal Regression)
    """

    def __init__(self):
        super(RetinaModel, self).__init__()

        # 1. Backbone
        # Load pre-trained ConvNeXt-Base.
        # num_classes=0 removes the default classifier.
        # global_pool="" removes the default pooling, returning feature maps.
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=Config.pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.drop_path_rate,
        )

        # Get the number of features output by the backbone
        # For ConvNeXt-Base, this is typically 1024
        self.num_features = self.backbone.num_features

        # 2. Pooling Layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM(p=3)

        # 3. Head
        # We concatenate GAP and GeM features, so input dim is doubled
        head_dim = self.num_features * 2

        self.neck = nn.LayerNorm(head_dim)

        # Multi-Sample Dropout
        # We use 5 dropout layers with p=0.5 (standard for MSD)
        # If Config.drop_rate is 0, we can default to 0.5 for MSD effectiveness,
        # or strictly follow config. Given the specific requirement for MSD,
        # using a fixed rate like 0.5 is standard practice even if global drop_rate is low.
        # However, to respect the Config class provided, we'll use a hardcoded 0.5
        # as MSD is a specific regularization technique distinct from standard dropout.
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])

        # Linear Layer for Ordinal Regression
        # Outputs 4 logits corresponding to P(y>0), P(y>1), P(y>2), P(y>3)
        self.fc = nn.Linear(head_dim, Config.num_classes)

        # Initialize weights for the head
        self._init_weights(self.fc)
        self._init_weights(self.neck)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # Extract features from backbone
        # Shape: [B, C, H, W]
        features = self.backbone(x)

        # Dual-Stream Pooling
        # GAP -> [B, C, 1, 1] -> [B, C]
        x_gap = self.global_pool(features).flatten(1)
        # GeM -> [B, C, 1, 1] -> [B, C]
        x_gem = self.gem_pool(features).flatten(1)

        # Concatenate
        # Shape: [B, 2*C]
        x = torch.cat([x_gap, x_gem], dim=1)

        # LayerNorm
        x = self.neck(x)

        # Multi-Sample Dropout & Linear
        # Average the predictions from multiple dropout masks
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))

        out /= len(self.dropouts)

        return out
