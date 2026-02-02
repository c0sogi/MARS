import torch
import torch.nn as nn
import timm
from library.config import Config


class SpatialSymmetryDifferenceNet(nn.Module):
    """
    Siamese Network with Spatial Symmetry-Difference Fusion.

    This model takes a pair of images (Target and Contralateral) and explicitly
    computes the feature-level difference to suppress symmetric tissue patterns
    (like breast density and age-related atrophy) while highlighting asymmetric
    anomalies (malignant lesions).
    """

    def __init__(self):
        super(SpatialSymmetryDifferenceNet, self).__init__()

        # 1. Backbone (Siamese Branch)
        # We use EfficientNet-B2.
        # num_classes=0 and global_pool='' ensures we get the spatial feature maps
        # (B, C, H', W') instead of a pooled vector.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.NUM_CHANNELS,
            num_classes=0,
            global_pool="",
            drop_rate=Config.DROP_RATE,
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Dynamically determine the number of output channels from the backbone
        # This makes the class robust to backbone changes in Config.
        # We run a dummy forward pass to get the shape.
        dummy_input = torch.zeros(1, Config.NUM_CHANNELS, 256, 256)
        with torch.no_grad():
            dummy_features = self.backbone(dummy_input)

        self.backbone_channels = dummy_features.shape[1]

        # 2. Fusion Head Configuration
        # The input to the head is the concatenation of the Target Map and the Difference Map.
        # Input Channels = Backbone Channels * 2
        head_in_channels = self.backbone_channels * 2

        # 3. Depthwise Separable Convolution Block
        # Reduces computation while mixing spatial and channel information from the fused map.
        self.head_conv = nn.Sequential(
            # Depthwise: Spatial filtering per channel
            nn.Conv2d(
                head_in_channels,
                head_in_channels,
                kernel_size=3,
                padding=1,
                groups=head_in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(head_in_channels),
            nn.SiLU(inplace=True),
            # Pointwise: Channel mixing and projection back to original dimension
            nn.Conv2d(
                head_in_channels, self.backbone_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.backbone_channels),
            nn.SiLU(inplace=True),
        )

        # 4. Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=Config.DROP_RATE)
        self.classifier = nn.Linear(self.backbone_channels, 1)

    def forward(self, target_img, contra_img):
        """
        Args:
            target_img (Tensor): The candidate breast image (B, C, H, W).
            contra_img (Tensor): The contralateral breast image (B, C, H, W).

        Returns:
            logits (Tensor): Raw output scores (B,).
        """
        # 1. Feature Extraction (Shared Weights)
        # Extract spatial maps: (B, C, H', W')
        target_features = self.backbone(target_img)
        contra_features = self.backbone(contra_img)

        # 2. Spatial Difference Operation
        # Subtract contralateral features from target features.
        # This cancels out symmetric signals (age, density) and highlights asymmetry.
        diff_features = target_features - contra_features

        # 3. Feature Fusion
        # Concatenate original target context with the difference signal.
        # Shape: (B, 2*C, H', W')
        fused_features = torch.cat([target_features, diff_features], dim=1)

        # 4. Head Processing
        # Apply Depthwise Separable Conv to refine features
        x = self.head_conv(fused_features)

        # Global Pooling -> (B, C, 1, 1)
        x = self.global_pool(x)

        # Flatten -> (B, C)
        x = x.flatten(1)

        # Dropout and Classification
        x = self.dropout(x)
        logits = self.classifier(x)

        # Cite debug_lesson_12: Return (B, 1) to match target (B, 1)
        return logits
