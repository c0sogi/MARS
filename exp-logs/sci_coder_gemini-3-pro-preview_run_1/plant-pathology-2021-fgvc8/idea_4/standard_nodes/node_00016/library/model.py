import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.

    Computes the generalized mean: f(X) = (1/|X| * sum(x^p))^(1/p)
    where p is a trainable parameter.

    - p=1 -> Average Pooling
    - p=infinity -> Max Pooling
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x is (B, C, H, W)
        # clamp min value to eps to avoid numerical instability with pow
        x = x.clamp(min=eps)

        # Calculate average of x^p over the spatial dimensions (H, W)
        # F.avg_pool2d with kernel_size=(H, W) computes the mean
        x_pow_p = x.pow(p)
        avg_x_pow_p = F.avg_pool2d(x_pow_p, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        gem_out = avg_x_pow_p.pow(1.0 / p)

        return gem_out

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class AppleConvNeXt(nn.Module):
    """
    ConvNeXt-Small model adapted for Apple Disease Detection.

    Features:
    - Backbone: convnext_small.in12k_ft_in1k (ImageNet-21k pre-trained)
    - Pooling: Generalized Mean Pooling (GeM) to focus on disease lesions
    - Head: BatchNorm -> Dropout -> Linear
    """

    def __init__(self, pretrained=True):
        super(AppleConvNeXt, self).__init__()

        # Load backbone
        # num_classes=0 and global_pool='' ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Get the number of input features for the head
        in_features = self.backbone.num_features

        # Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM(p=Config.GEM_P)
        else:
            # Fallback to standard Adaptive Average Pooling if GeM is disabled
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(Config.DROP_RATE),
            nn.Linear(in_features, Config.NUM_CLASSES),
        )

        # Initialize weights for the head
        self._init_weights(self.head)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. Feature Extraction (B, C, H, W)
        features = self.backbone(x)

        # 2. Pooling (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # 3. Flatten (B, C)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # 4. Classification Head (B, Num_Classes)
        logits = self.head(flattened_features)

        return logits
