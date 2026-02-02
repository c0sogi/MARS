import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp min value to eps to avoid numerical instability with pow
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Apply average pooling on x^p, then take the (1/p)-th root
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


class WhaleEfficientNetV2(nn.Module):
    """
    Whale Call Detection Model based on EfficientNetV2-Medium.

    Architecture:
    1. Backbone: EfficientNetV2-M (ImageNet-21k pretrained)
    2. Pooling: Generalized Mean (GeM) Pooling
    3. Head: Dropout + Linear Layer
    """

    def __init__(self, pretrained=True):
        super(WhaleEfficientNetV2, self).__init__()

        # Initialize Backbone using timm
        # num_classes=0 removes the classification head
        # global_pool="" removes the default pooling, returning spatial feature maps
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Retrieve the number of output features from the backbone
        in_features = self.backbone.num_features

        # Pooling Layer
        self.pooling = GeM()

        # Classification Head
        self.drop = nn.Dropout(Config.DROPOUT_RATE)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of shape (Batch, 3, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, 1)
        """
        # 1. Feature Extraction
        features = self.backbone(x)  # Output: (Batch, Channels, H_feat, W_feat)

        # 2. Pooling
        pooled = self.pooling(features)  # Output: (Batch, Channels, 1, 1)

        # 3. Flatten
        flattened = pooled.view(pooled.size(0), -1)  # Output: (Batch, Channels)

        # 4. Classification Head
        dropped = self.drop(flattened)
        logits = self.fc(dropped)  # Output: (Batch, 1)

        return logits
