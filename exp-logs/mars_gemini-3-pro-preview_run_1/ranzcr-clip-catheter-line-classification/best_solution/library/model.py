import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Learns a parameter p to interpolate between Max Pooling (p -> infinity)
    and Average Pooling (p -> 1).
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid numerical instability with pow
        x = x.clamp(min=eps)

        # Calculate average of x^p
        # F.avg_pool2d with kernel size equal to input spatial dimensions acts as global average pooling
        x_pow = x.pow(p)
        avg_pool = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        return avg_pool.pow(1.0 / p)


class CatheterModel(nn.Module):
    """
    EfficientNet-V2-S based model with GeM pooling for Catheter Detection.
    """

    def __init__(
        self,
        model_name=Config.model_name,
        pretrained=Config.pretrained,
        num_classes=Config.num_classes,
        in_channels=Config.in_channels,
        drop_path_rate=Config.drop_path_rate,
    ):
        super(CatheterModel, self).__init__()

        # Load backbone from timm
        # num_classes=0 and global_pool='' ensures we get the spatial feature maps
        # instead of the pooled vector or logits.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
            drop_path_rate=drop_path_rate,
        )

        # Retrieve the number of output features from the backbone
        self.in_features = self.backbone.num_features

        # Replace standard pooling with GeM pooling
        # This helps in preserving high activations from thin structures like catheters
        self.pooling = GeM()

        # Final classification head
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        # Extract spatial features: (B, C, H, W)
        features = self.backbone(x)

        # Apply GeM pooling: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # Classification logits: (B, num_classes)
        logits = self.fc(pooled_features)

        return logits
