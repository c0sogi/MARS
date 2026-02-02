import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the spatial dimensions of the input tensor.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6, trainable=True):
        super(GeM, self).__init__()
        # p can be a learnable parameter or fixed
        if trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        # Ensure numerical stability by clamping input
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Apply average pooling on x^p, then raise to 1/p
        # Kernel size matches the spatial dimensions (H, W)
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


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model using ConvNeXt backbone and GeM pooling.
    """

    def __init__(self, cfg=Config):
        super(AppleDiseaseModel, self).__init__()

        # Initialize backbone using timm
        # num_classes=0 and global_pool='' ensures we get the feature map (B, C, H, W)
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=cfg.PRETRAINED,
            drop_path_rate=cfg.DROP_PATH_RATE,
            num_classes=0,
            global_pool="",
        )

        # Get the number of output features from the backbone
        in_features = self.backbone.num_features

        # Initialize Pooling Layer
        if cfg.POOLING == "gem":
            self.pooling = GeM(p=cfg.GEM_P, trainable=cfg.GEM_TRAINABLE)
        else:
            # Fallback to standard Global Average Pooling
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.head = nn.Linear(in_features, cfg.NUM_CLASSES)

    def forward(self, x):
        # Extract features from backbone
        # Shape: (Batch_Size, Channels, Height, Width)
        features = self.backbone(x)

        # Apply Pooling
        # Shape: (Batch_Size, Channels, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten
        # Shape: (Batch_Size, Channels)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Classification
        # Shape: (Batch_Size, Num_Classes)
        logits = self.head(flattened_features)

        return logits
