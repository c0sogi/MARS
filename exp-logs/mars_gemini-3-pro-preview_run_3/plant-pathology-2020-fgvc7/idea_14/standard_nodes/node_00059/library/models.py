import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a parameter 'p' to interpolate between Average Pooling (p=1) and Max Pooling (p=infinity).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp inputs to eps to avoid NaN gradients
        # Average pool over spatial dimensions (H, W)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class AppleNet(nn.Module):
    """
    Apple Disease Detection Model.

    Architecture:
    1. Backbone: timm model (EfficientNetV2 or MaxViT) in features_only mode.
    2. Neck: Feature Pyramid Network (FPN) fusing last 3 stages.
    3. Pooling: Independent GeM pooling for each FPN level.
    4. Head: Decoupled Multi-Task Head (Main Softmax + Aux Binary Sigmoids).
    """

    def __init__(self, model_name, pretrained=True):
        super(AppleNet, self).__init__()

        # 1. Backbone
        # Initialize with features_only=True to access intermediate layers
        try:
            self.backbone = timm.create_model(
                model_name, pretrained=pretrained, features_only=True
            )
        except RuntimeError as e:
            if "Unknown model" in str(e):
                print(f"ERROR: Model '{model_name}' not found in timm registry.")
                try:
                    matches = timm.list_models("*efficientnetv2*")
                    print(f"Available EfficientNetV2 models: {matches[:20]} ...")
                except Exception:
                    pass
            raise e

        # Extract channels from the last 3 stages
        # This dynamic slicing ensures compatibility between EfficientNet (5 stages) and MaxViT (4 stages)
        all_feature_channels = self.backbone.feature_info.channels()
        if len(all_feature_channels) < 3:
            raise ValueError(f"Backbone {model_name} has fewer than 3 stages.")

        # We take the last 3 available feature maps
        self.feature_channels = all_feature_channels[-3:]

        # 2. FPN Lateral Layers
        # Project different channel depths to a common FPN dimension
        self.fpn_channels = Config.FPN_OUT_CHANNELS

        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_c, self.fpn_channels, kernel_size=1)
                for in_c in self.feature_channels
            ]
        )

        # 3. GeM Pooling
        # Learnable pooling parameter initialized to Config.GEM_P
        self.gem = GeM(p=Config.GEM_P)

        # 4. Heads
        # We concatenate pooled features from 3 FPN levels
        # Input dim = FPN_channels * 3
        head_in_features = self.fpn_channels * 3

        self.dropout = nn.Dropout(p=Config.DROP_RATE)

        # Main Head: 4 classes (Healthy, Multiple, Rust, Scab)
        self.fc_main = nn.Linear(head_in_features, Config.NUM_CLASSES)

        # Aux Head A: Has Rust (Binary)
        self.fc_rust = nn.Linear(head_in_features, 1)

        # Aux Head B: Has Scab (Binary)
        self.fc_scab = nn.Linear(head_in_features, 1)

        # Initialize weights for heads
        self._init_weights(self.lateral_convs)
        self._init_weights(self.fc_main)
        self._init_weights(self.fc_rust)
        self._init_weights(self.fc_scab)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.ModuleList):
            for m in module:
                self._init_weights(m)

    def forward(self, x):
        # 1. Backbone Features
        # Get all features and slice the last 3
        features = self.backbone(x)[-3:]

        # Unpack: [Smallest Spatial, Medium Spatial, Largest Spatial]
        # c3: High Res / Low Semantics
        # c5: Low Res / High Semantics
        c3, c4, c5 = features

        # 2. FPN Logic
        # Lateral projections (1x1 convs)
        p5 = self.lateral_convs[2](c5)
        p4_lat = self.lateral_convs[1](c4)
        p3_lat = self.lateral_convs[0](c3)

        # Top-down pathway with nearest neighbor upsampling
        # P5 is the top level
        # P4 = L4 + Upsample(P5)
        p4 = p4_lat + F.interpolate(p5, size=p4_lat.shape[-2:], mode="nearest")

        # P3 = L3 + Upsample(P4)
        p3 = p3_lat + F.interpolate(p4, size=p3_lat.shape[-2:], mode="nearest")

        # 3. Pooling & Concatenation
        # Apply GeM to each FPN level independently
        # Flatten to (Batch, Channels)
        v5 = self.gem(p5).flatten(1)
        v4 = self.gem(p4).flatten(1)
        v3 = self.gem(p3).flatten(1)

        # Concatenate features: (Batch, FPN_Channels * 3)
        fused_features = torch.cat([v3, v4, v5], dim=1)

        # 4. Heads
        fused_features = self.dropout(fused_features)

        logits_main = self.fc_main(fused_features)
        logits_rust = self.fc_rust(fused_features)
        logits_scab = self.fc_scab(fused_features)

        # Return dictionary for multi-task loss
        # Main head uses CrossEntropy (expects raw logits)
        # Aux heads use BCEWithLogits (expects raw logits)
        return {"main": logits_main, "rust": logits_rust, "scab": logits_scab}
