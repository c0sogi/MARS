import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import library.config as cfg


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean of each channel in the feature map.
    f(X) = (1/|X| * sum(x^p))^(1/p)

    The parameter p is learnable.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # Initialize p as a learnable parameter, typically starting at 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # Clamp input to be positive (epsilon) to avoid NaN gradients with pow
        # Apply average pooling on x^p, then take the (1/p)-th root
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class RetinopathyModel(nn.Module):
    """
    Retinopathy Severity Classifier/Regressor.

    Backbone: EfficientNet-B5 (Noisy Student weights)
    Pooling: GeM (Generalized Mean Pooling)
    Head: Linear Layer (Regression)
    """

    def __init__(self, pretrained=True):
        super(RetinopathyModel, self).__init__()

        # Create backbone using timm
        # num_classes=0 removes the fully connected layer
        # global_pool='' removes the default pooling layer, returning feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            cfg.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=cfg.DROP_PATH_RATE,
        )

        # Get the number of output features from the backbone
        self.in_features = self.backbone.num_features

        # Pooling Layer
        if cfg.USE_GEM_POOLING:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Dropout and Classification Head
        self.dropout = nn.Dropout(p=cfg.DROPOUT_RATE)
        self.fc = nn.Linear(self.in_features, cfg.NUM_CLASSES)

    def forward(self, x):
        # Extract features: (B, C, H, W)
        x = self.backbone(x)

        # Apply Pooling: (B, C, 1, 1)
        x = self.pool(x)

        # Flatten: (B, C)
        x = x.flatten(1)

        # Apply Dropout
        x = self.dropout(x)

        # Regression Head: (B, 1)
        x = self.fc(x)

        return x
