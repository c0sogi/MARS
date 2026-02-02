import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import ModelConfig


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN in pow() and ensure numerical stability
        x = x.clamp(min=self.eps)

        # Calculate GeM: (Avg(x^p))^(1/p)
        # We use avg_pool2d to compute the mean over spatial dimensions (H, W)
        x_pow = x.pow(self.p)
        avg_x_pow = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))
        return avg_x_pow.pow(1.0 / self.p)


class WhaleEfficientNet(nn.Module):
    """
    EfficientNetV2-Medium backbone with GeM pooling and a linear classification head.
    Designed for high-resolution spectrogram inputs.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super(WhaleEfficientNet, self).__init__()

        # Initialize the backbone using timm
        # in_chans=1: Adapts the first conv layer for 1-channel (grayscale) spectrograms
        # num_classes=0, global_pool='': Returns unpooled spatial feature maps
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            in_chans=config.in_channels,
            num_classes=0,
            global_pool="",
            drop_path_rate=config.drop_path_rate,
        )

        # Determine the number of output features from the backbone
        self.num_features = self.backbone.num_features

        # Pooling Layer
        if config.pool_type.lower() == "gem":
            self.pool = GeM()
        else:
            # Fallback to standard Global Average Pooling
            self.pool = nn.AdaptiveAvgPool2d(1)

        # Regularization
        self.drop = nn.Dropout(p=config.dropout_rate)

        # Classification Head
        # Projects features to the number of classes (1 for binary classification)
        self.fc = nn.Linear(self.num_features, config.num_classes)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x: Input tensor of shape (Batch, 1, F, T)
        Returns:
            Logits of shape (Batch, num_classes)
        """
        # 1. Feature Extraction
        # Output shape: (Batch, num_features, H_feat, W_feat)
        features = self.backbone(x)

        # 2. Pooling
        # Output shape: (Batch, num_features, 1, 1)
        pooled = self.pool(features)

        # 3. Flatten
        # Output shape: (Batch, num_features)
        flattened = pooled.flatten(1)

        # 4. Dropout
        dropped = self.drop(flattened)

        # 5. Classification
        # Output shape: (Batch, num_classes)
        logits = self.fc(dropped)

        return logits
