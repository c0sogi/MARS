import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes f = (mean(x^p))^(1/p).

    This pooling layer is learnable and interpolates between Average Pooling (p=1)
    and Max Pooling (p -> infinity). It is particularly effective for retrieving
    sparse features (like 'needles') from a noisy background.
    """

    def __init__(self, p=3.0, eps=1e-6):
        """
        Args:
            p (float): Initial power parameter.
            eps (float): Small constant to avoid numerical instability.
        """
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # We clamp the input to min=eps to ensure non-negative values before raising to power p.
        # This acts similarly to a ReLU activation and prevents NaN gradients.
        # F.avg_pool2d calculates the mean over the spatial dimensions (H, W).
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class SETIModel(nn.Module):
    """
    SETI Signal Detection Model.

    Structure:
    1. Backbone: EfficientNet-V2 Small (pretrained on ImageNet)
       - Extracts hierarchical features from the vertical stack of spectrograms.
    2. Pooling: GeM (Generalized Mean Pooling)
       - Aggregates spatial features, focusing on high-activation regions (signals).
    3. Head: Linear Classifier
       - Maps the feature vector to binary logits.
    """

    def __init__(self, pretrained=True):
        """
        Args:
            pretrained (bool): Whether to load ImageNet weights for the backbone.
        """
        super(SETIModel, self).__init__()

        # Create backbone using timm
        # num_classes=0 removes the default classifier head
        # global_pool='' removes the default pooling, returning the spatial feature map
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=pretrained,
            in_chans=Config.in_channels,
            num_classes=0,
            global_pool="",
        )

        # Retrieve the number of output channels from the backbone
        self.in_features = self.backbone.num_features

        # Initialize Pooling Layer
        if Config.use_gem:
            self.pooling = GeM(p=Config.gem_p)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Initialize Classifier Head
        self.drop = nn.Dropout(Config.drop_rate)
        self.fc = nn.Linear(self.in_features, Config.num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes).
        """
        # 1. Feature Extraction
        # Output shape: (Batch, Features, H', W')
        features = self.backbone(x)

        # 2. Pooling
        # GeM reduces spatial dimensions to 1x1
        # Output shape: (Batch, Features, 1, 1)
        pooled = self.pooling(features)

        # 3. Flatten
        # Output shape: (Batch, Features)
        flattened = pooled.view(pooled.size(0), -1)

        # 4. Dropout
        dropped = self.drop(flattened)

        # 5. Classification
        # Output shape: (Batch, num_classes)
        logits = self.fc(dropped)

        return logits
