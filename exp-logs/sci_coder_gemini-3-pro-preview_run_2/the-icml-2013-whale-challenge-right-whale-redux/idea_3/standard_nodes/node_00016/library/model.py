import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN with log/pow operations
        x = x.clamp(min=eps)
        # Apply GeM formula: (AvgPool(x^p))^(1/p)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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


class WhaleConvNeXt(nn.Module):
    """
    ConvNeXt model adapted for 1-channel spectrogram input with GeM pooling.
    """

    def __init__(self):
        super(WhaleConvNeXt, self).__init__()

        # 1. Load Pretrained Backbone
        # Use in_chans=Config.IN_CHANNELS (1) so timm automatically adapts the first layer weights
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=Config.IN_CHANNELS,
        )

        # 3. Define Pooling
        if Config.USE_GEM_POOLING:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # 4. Define Classifier Head
        self.num_features = self.backbone.num_features
        self.head = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.head(x)
        return x
