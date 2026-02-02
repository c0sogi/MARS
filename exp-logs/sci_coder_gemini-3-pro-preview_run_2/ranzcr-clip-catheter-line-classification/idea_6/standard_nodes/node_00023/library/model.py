import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (AvgPool(x^p))^(1/p).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            return self.gem(x.float(), p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp input for numerical stability before power operation
        x = x.clamp(min=eps)
        # Apply power p
        x = x.pow(p)
        # Average pooling over the spatial dimensions (H, W)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        # Apply root 1/p
        x = x.pow(1.0 / p)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", eps="
            + str(self.eps)
            + ")"
        )


class CatheterModel(nn.Module):
    """
    Catheter Detection Model using ConvNeXt V2 Tiny backbone with Dual-Stage GeM Pooling.
    """

    def __init__(self):
        super(CatheterModel, self).__init__()

        # --- Backbone ---
        # Load ConvNeXt V2 Tiny with features_only=True to access intermediate layers.
        # We do not specify out_indices explicitly here to avoid indexing conflicts;
        # instead we select the necessary stages from the full list of features.
        self.backbone = timm.create_model(
            Config.model_name, pretrained=True, features_only=True
        )

        # --- Feature Dimensions ---
        # Get channel counts for the feature maps.
        # ConvNeXt V2 Tiny typically returns features with channels: [96, 192, 384, 768]
        feature_channels = self.backbone.feature_info.channels()

        # We require the last two stages (Stage 3 and Stage 4) for dual-stage pooling
        # as per the strategy (384 + 768 = 1152).
        self.dim_stage3 = feature_channels[-2]
        self.dim_stage4 = feature_channels[-1]

        self.head_in_features = self.dim_stage3 + self.dim_stage4

        # --- Pooling ---
        # Independent GeM pooling layers for each stage to preserve scale-specific details
        self.gem_stage3 = GeM()
        self.gem_stage4 = GeM()

        # --- Head ---
        self.drop_rate = Config.fc_dropout
        self.fc = nn.Linear(self.head_in_features, Config.num_classes)

    def forward(self, x):
        # 1. Backbone Feature Extraction
        features = self.backbone(x)

        # 2. Select Stages
        # Extract the penultimate (Stage 3) and final (Stage 4) feature maps
        feat_stage3 = features[-2]  # Shape: (B, 384, H/16, W/16)
        feat_stage4 = features[-1]  # Shape: (B, 768, H/32, W/32)

        # 3. Dual-Stage GeM Pooling
        # Pool each stage independently. Output shape: (B, C, 1, 1)
        pool_stage3 = self.gem_stage3(feat_stage3).flatten(1)
        pool_stage4 = self.gem_stage4(feat_stage4).flatten(1)

        # 4. Concatenation
        # Combine semantic context (Stage 4) with spatial detail (Stage 3)
        global_features = torch.cat([pool_stage3, pool_stage4], dim=1)

        # 5. Multi-Sample Dropout Head
        if self.training:
            # During training, compute logits multiple times with different dropout masks
            # and average them to improve generalization and convergence speed.
            logits_list = []
            for _ in range(5):
                dropped = F.dropout(global_features, p=self.drop_rate, training=True)
                logits_list.append(self.fc(dropped))
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            # During inference, perform a single forward pass
            logits = self.fc(global_features)

        return logits
