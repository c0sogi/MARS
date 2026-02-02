import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a parameter 'p' to transition between Average Pooling (p=1) and Max Pooling (p=infinity).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp for numerical stability
        x = x.clamp(min=eps)
        # Average pool of x^p
        x_pow = x.pow(p)
        pooled = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        # (Average)^ (1/p)
        return pooled.pow(1.0 / p)


class RetinopathyModel(nn.Module):
    """
    Diabetic Retinopathy Classification Model.
    Architecture: ConvNeXt-Small -> GeM Pooling -> Multi-Sample Dropout -> Ordinal Linear Head.
    """

    def __init__(self):
        super(RetinopathyModel, self).__init__()

        # 1. Backbone
        # We disable the built-in head (num_classes=0) and pooling (global_pool="")
        # to get the spatial feature map for GeM.
        self.backbone = timm.create_model(
            Config.backbone,
            pretrained=Config.pretrained,
            in_chans=Config.in_chans,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.backbone_drop_path_rate,
        )

        # Determine feature dimension dynamically
        # Run a dummy forward pass with a small input to get channel count
        with torch.no_grad():
            dummy_input = torch.zeros(1, Config.in_chans, 64, 64)
            features = self.backbone(dummy_input)
            self.in_features = features.shape[1]

        # 2. Pooling Mechanism
        if Config.use_gem_pooling:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # 3. Classification Head
        self.use_msd = Config.use_multi_sample_dropout
        self.num_heads = Config.num_ordinal_heads

        if self.use_msd:
            # Create 5 dropout layers with the configured rate
            self.dropouts = nn.ModuleList(
                [nn.Dropout(Config.dropout_rate) for _ in range(5)]
            )
        else:
            self.dropout = nn.Dropout(Config.dropout_rate)

        # Final Linear Layer for Ordinal Regression
        # Outputs 4 values: P(y>=1), P(y>=2), P(y>=3), P(y>=4)
        self.fc = nn.Linear(self.in_features, self.num_heads)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
        Returns:
            torch.Tensor: Logits (B, num_ordinal_heads)
        """
        # Feature Extraction
        features = self.backbone(x)  # (B, C, H_feat, W_feat)

        # Pooling
        features = self.pooling(features)  # (B, C, 1, 1)
        features = features.flatten(1)  # (B, C)

        # Head (Dropout + Linear)
        if self.use_msd:
            # Multi-Sample Dropout: Average the logits from multiple dropout masks
            logits_list = []
            for dropout_layer in self.dropouts:
                dropped_features = dropout_layer(features)
                logits_list.append(self.fc(dropped_features))

            # Stack and average
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            # Standard Dropout
            features = self.dropout(features)
            logits = self.fc(features)

        return logits
