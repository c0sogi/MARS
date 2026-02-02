import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (AvgPool(x^p))^(1/p) where p is a learnable parameter.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp for numerical stability before power operation
        x = x.clamp(min=eps)
        # Apply power p, average pool over spatial dimensions, then root p
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class CatheterModel(nn.Module):
    """
    Multi-Scale Feature Aggregation Model for Catheter Detection.
    Uses EfficientNetV2-S backbone with GeM pooling on strides 8, 16, and 32.
    Includes an auxiliary segmentation head for background suppression.
    """

    def __init__(self, pretrained=True):
        super(CatheterModel, self).__init__()

        # 1. Backbone: EfficientNetV2-S
        # Extract features at indices 2, 3, 4 corresponding to strides 8, 16, 32
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        # Determine channel counts dynamically
        # Create a dummy input to trace shapes
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy_input)
        self.feature_channels = [f.shape[1] for f in feats]

        # 2. Classification Head Components
        # Independent GeM pooling for each feature scale
        self.gems = nn.ModuleList([GeM(p=3.0) for _ in self.feature_channels])

        # Final Linear Layer
        # Input dimension is the sum of channels from all scales
        total_channels = sum(self.feature_channels)
        self.drop = nn.Dropout(Config.DROP_RATE)
        self.fc = nn.Linear(total_channels, Config.NUM_CLASSES)

        # 3. Auxiliary Segmentation Head Components
        # Project all scales to a common dimension for fusion
        aux_dim = 64
        self.aux_convs = nn.ModuleList(
            [nn.Conv2d(in_c, aux_dim, kernel_size=1) for in_c in self.feature_channels]
        )

        # Fusion and Prediction
        # Input: 3 scales * aux_dim
        self.aux_classifier = nn.Sequential(
            nn.Conv2d(aux_dim * 3, aux_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(aux_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(aux_dim, 1, kernel_size=1),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)

        Returns:
            logits (torch.Tensor): Classification logits (B, NumClasses)
            mask (torch.Tensor): Auxiliary segmentation mask (B, 1, H, W)
        """
        input_size = x.shape[-2:]

        # Extract multi-scale features
        # feats[0]: Stride 8
        # feats[1]: Stride 16
        # feats[2]: Stride 32
        feats = self.backbone(x)

        # --- Classification Path ---
        pooled_feats = []
        for i, feat in enumerate(feats):
            # Apply GeM -> (B, C, 1, 1)
            pooled = self.gems[i](feat)
            # Flatten -> (B, C)
            pooled = pooled.flatten(1)
            pooled_feats.append(pooled)

        # Concatenate features from all scales
        concat_feats = torch.cat(pooled_feats, dim=1)

        # Dropout and Classify
        embedding = self.drop(concat_feats)
        logits = self.fc(embedding)

        # --- Auxiliary Segmentation Path ---
        # Target spatial resolution is that of the largest feature map (Stride 8)
        target_h, target_w = feats[0].shape[-2:]

        aux_feats = []
        for i, feat in enumerate(feats):
            # Project channels to aux_dim
            proj = self.aux_convs[i](feat)

            # Upsample lower resolution maps (Stride 16, 32) to Stride 8
            if proj.shape[-2:] != (target_h, target_w):
                proj = F.interpolate(
                    proj,
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                )
            aux_feats.append(proj)

        # Fuse features (Concatenate along channel dim)
        aux_concat = torch.cat(aux_feats, dim=1)

        # Predict mask at Stride 8
        aux_out = self.aux_classifier(aux_concat)

        # Upsample mask to original image resolution
        mask = F.interpolate(
            aux_out, size=input_size, mode="bilinear", align_corners=False
        )

        return logits, mask
