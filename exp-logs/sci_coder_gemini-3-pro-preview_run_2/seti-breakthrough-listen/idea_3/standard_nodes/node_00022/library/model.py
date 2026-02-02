import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer for 2D feature maps.
    Computes the p-norm of the input tensor.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Pools over spatial dimensions (H, W)
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class SETIModel(nn.Module):
    """
    EfficientNetV2-S with GeM Pooling for SETI signal detection.
    Treats the vertically stacked panels as a single 2D image.
    """

    def __init__(self, pretrained=True):
        super(SETIModel, self).__init__()

        # Initialize Backbone
        # num_classes=0 and global_pool='' ensures we get the feature map
        self.backbone = timm.create_model(
            Config.model_name,
            pretrained=pretrained,
            in_chans=Config.in_channels,
            num_classes=0,
            global_pool="",
        )

        # Dynamically determine the number of input features
        dummy_input = torch.zeros(1, Config.in_channels, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # Generalized Mean Pooling
        self.gem = GeM(p=3.0)

        # Classification Head
        self.drop = nn.Dropout(Config.drop_rate)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 3, 1638, 256)
        Returns:
            logits: Output tensor of shape (B, 1)
        """
        # 1. Backbone Feature Extraction
        # Shape: (B, C, H_feat, W_feat)
        x = self.backbone(x)

        # 2. GeM Pooling
        # Shape: (B, C, 1, 1)
        x = self.gem(x)

        # Flatten: (B, C)
        x = x.view(x.size(0), -1)

        # 3. Classifier
        x = self.drop(x)
        logits = self.fc(x)

        return logits
