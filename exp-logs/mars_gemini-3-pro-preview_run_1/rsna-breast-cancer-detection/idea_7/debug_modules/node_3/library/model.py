import torch
import torch.nn as nn
import timm
from library.config import Config


class SpatialSiameseEfficientNet(nn.Module):
    """
    Spatial Symmetry-Difference Siamese Network based on EfficientNet-B2.

    Architecture:
    1. Shared Backbone (EfficientNet-B2) extracts spatial feature maps.
    2. Spatial Difference: M_diff = M_target - M_contra.
    3. Fusion: Concat(M_target, M_diff).
    4. Head: Depthwise Separable Conv -> Global Pool -> Linear.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone
        # We use forward_features to get spatial maps, so num_classes is irrelevant here
        # but we set it to 0 to avoid creating the default classifier head in the backbone.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
            num_classes=0,
        )

        # Determine the number of output channels from the backbone
        # We run a dummy forward pass to dynamically retrieve the shape.
        dummy_input = torch.randn(1, Config.IN_CHANNELS, 256, 256)
        with torch.no_grad():
            # forward_features returns the last feature map (B, C, H, W)
            features = self.backbone.forward_features(dummy_input)

        self.num_features = features.shape[1]

        # 2. Fusion Head
        # Input channels = num_features (Target) + num_features (Diff) = 2 * num_features

        # Depthwise Convolution: Spatial mixing per channel
        self.dw_conv = nn.Conv2d(
            in_channels=self.num_features * 2,
            out_channels=self.num_features * 2,
            kernel_size=3,
            padding=1,
            groups=self.num_features * 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.num_features * 2)
        self.act1 = nn.SiLU(inplace=True)

        # Pointwise Convolution: Channel mixing and reduction
        self.pw_conv = nn.Conv2d(
            in_channels=self.num_features * 2,
            out_channels=self.num_features,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.num_features)
        self.act2 = nn.SiLU(inplace=True)

        # 3. Classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=Config.DROP_RATE)
        self.classifier = nn.Linear(self.num_features, 1)

    def forward_features_map(self, x):
        """Extracts spatial feature maps from the backbone."""
        return self.backbone.forward_features(x)

    def forward(self, x, x_contra):
        """
        Args:
            x (torch.Tensor): Target images (B, C, H, W)
            x_contra (torch.Tensor): Contralateral images (B, C, H, W)

        Returns:
            torch.Tensor: Logits (B, 1)
        """
        # 1. Extract Feature Maps
        # Shape: (B, num_features, H', W')
        f_target = self.forward_features_map(x)
        f_contra = self.forward_features_map(x_contra)

        # 2. Spatial Symmetry-Difference
        # Subtract contralateral features from target features
        # This suppresses symmetric background tissue patterns
        f_diff = f_target - f_contra

        # 3. Concatenation
        # Stack target features (context) and difference features (anomaly signal)
        # Shape: (B, 2 * num_features, H', W')
        f_fused = torch.cat([f_target, f_diff], dim=1)

        # 4. Head Processing
        # Depthwise Separable Conv Block
        out = self.dw_conv(f_fused)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.pw_conv(out)
        out = self.bn2(out)
        out = self.act2(out)

        # Pooling and Classification
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)

        logits = self.classifier(out)

        return logits
