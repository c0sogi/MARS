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
    Catheter Detection Model using EfficientNetV2-S backbone with Single-Stage GeM Pooling.
    """

    def __init__(self):
        super(CatheterModel, self).__init__()

        # --- Backbone ---
        self.backbone = timm.create_model(
            Config.model_name, pretrained=True, features_only=True
        )

        # --- Feature Dimensions ---
        # EfficientNetV2-S last feature map has 1280 channels
        self.head_in_features = Config.backbone_dim

        # --- Pooling ---
        self.gem = GeM()

        # --- Head ---
        self.drop_rate = Config.fc_dropout
        self.fc = nn.Linear(self.head_in_features, Config.num_classes)

    def forward(self, x):
        # 1. Backbone Feature Extraction
        features = self.backbone(x)

        # 2. Select Last Stage
        last_feature = features[-1]

        # 3. GeM Pooling
        global_features = self.gem(last_feature).flatten(1)

        # 4. Multi-Sample Dropout Head
        if self.training:
            logits_list = []
            for _ in range(5):
                dropped = F.dropout(global_features, p=self.drop_rate, training=True)
                logits_list.append(self.fc(dropped))
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            logits = self.fc(global_features)

        return logits
