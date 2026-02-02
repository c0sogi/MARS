import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Focuses on high-activation spatial regions (lesions) more than Average Pooling.
    Formula: f(X) = (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp for numerical stability
        x = x.clamp(min=self.eps)

        # Average pooling on x^p
        # We use adaptive_avg_pool2d to handle variable sizes if necessary,
        # though here inputs are fixed size.
        x_pow = x.pow(self.p)
        pooled = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Take the p-th root
        return pooled.pow(1.0 / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.data.item():.4f}, eps={self.eps})"


class MultiSampleDropoutHead(nn.Module):
    """
    Classification head with Multi-Sample Dropout.
    Passes features through multiple dropout masks and averages the results
    to accelerate convergence and improve generalization.
    """

    def __init__(self, in_features, out_features, dropout_rates):
        super(MultiSampleDropoutHead, self).__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in dropout_rates])
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (Batch, In_Features)
        logits_list = []
        for dropout in self.dropouts:
            # Apply dropout then linear projection
            logits_list.append(self.fc(dropout(x)))

        # Stack and average the logits
        # Shape: (Batch, Out_Features)
        return torch.stack(logits_list, dim=0).mean(dim=0)


class AppleDiseaseModel(nn.Module):
    """
    Main model class for Apple Disease Detection.
    Integrates Backbone + GeM Pooling + Multi-Sample Dropout Head.
    """

    def __init__(self, model_name, pretrained=True):
        super(AppleDiseaseModel, self).__init__()

        # 1. Load Backbone
        # num_classes=0 and global_pool='' returns the raw spatial features (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input channels for the head
        self.in_features = self.backbone.num_features

        # 2. Pooling Layer
        if Config.USE_GEM_POOLING:
            self.global_pool = GeM(p=Config.GEM_P_INIT)
        else:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 3. Classification Head
        if Config.USE_MULTI_SAMPLE_DROPOUT:
            self.head = MultiSampleDropoutHead(
                in_features=self.in_features,
                out_features=Config.NUM_CLASSES,
                dropout_rates=Config.DROPOUT_RATES,
            )
        else:
            self.head = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        # 1. Feature Extraction
        # Shape: (B, C, H, W)
        features = self.backbone(x)

        # 2. Global Pooling
        # Shape: (B, C, 1, 1)
        features = self.global_pool(features)

        # Flatten: (B, C)
        features = features[:, :, 0, 0]

        # 3. Classification
        # Shape: (B, Num_Classes)
        logits = self.head(features)

        return logits
