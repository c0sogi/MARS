import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes (mean(x^p))^(1/p).
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN in pow if input is negative (e.g., from SiLU/Swish activations)
        # We avoid explicit casting to FP32 to maintain Mixed Precision performance where possible,
        # relying on PyTorch's stability.
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


class MultiTaskEfficientNet(nn.Module):
    """
    EfficientNetV2-S backbone with GeM Pooling.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        super().__init__()

        # 1. Backbone
        # features_only=True returns a list of feature maps from different stages
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        feature_channels = self.backbone.feature_info.channels()

        # 2. Classification Head
        # Uses the last feature map (typically stride 32)
        last_channel = feature_channels[-1]
        self.gem = GeM()
        self.drop = nn.Dropout(p=0.2)
        self.fc = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        # x: (B, 3, H, W)

        # Extract features from backbone
        features = self.backbone(x)

        # --- Classification Branch ---
        global_feat = features[-1]
        global_feat = self.gem(global_feat)
        global_feat = global_feat.view(global_feat.size(0), -1)
        global_feat = self.drop(global_feat)
        cls_logits = self.fc(global_feat)

        return cls_logits, None
