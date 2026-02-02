import torch
import torch.nn as nn
import timm
from library.config import (
    BACKBONE,
    PRETRAINED,
    DROP_RATE,
    DROP_PATH_RATE,
    IN_CHANNELS,
    IMG_SIZE,
)


class SpatialSiameseModel(nn.Module):
    """
    Spatial Symmetry-Difference Siamese Network.

    This model leverages the bilateral symmetry of breasts to detect anomalies.
    It computes a spatial difference map between the target breast and the
    contralateral breast at the feature map level, allowing the network to
    suppress symmetric background tissue and highlight local asymmetries (tumors).
    """

    def __init__(self):
        super(SpatialSiameseModel, self).__init__()

        # 1. Shared Backbone
        # We use features_only=False with num_classes=0 and global_pool=''
        # to get the final convolutional feature map (B, C, H, W).
        self.backbone = timm.create_model(
            BACKBONE,
            pretrained=PRETRAINED,
            in_chans=IN_CHANNELS,
            features_only=False,
            drop_rate=DROP_RATE,
            drop_path_rate=DROP_PATH_RATE,
            num_classes=0,
            global_pool="",
        )

        # 2. Determine Backbone Output Channels Dynamically
        # Run a dummy forward pass to get the feature map shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, IN_CHANNELS, IMG_SIZE[0], IMG_SIZE[1])
            features = self.backbone(dummy_input)
            in_features = features.shape[1]

        # The fusion concatenates the Target Map and the Difference Map
        fusion_channels = in_features * 2

        # 3. Fusion Head (Depthwise Separable Convolution)
        # Mixes the target texture features with the asymmetry signal.
        self.head = nn.Sequential(
            # Depthwise Convolution: Spatial mixing per channel
            nn.Conv2d(
                fusion_channels,
                fusion_channels,
                kernel_size=3,
                padding=1,
                groups=fusion_channels,
                bias=False,
            ),
            nn.BatchNorm2d(fusion_channels),
            nn.SiLU(inplace=True),
            # Pointwise Convolution: Channel mixing & reduction
            # Reduces channels back to in_features to control parameter count
            nn.Conv2d(fusion_channels, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.SiLU(inplace=True),
        )

        # 4. Classification Layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=DROP_RATE)
        self.fc = nn.Linear(in_features, 1)

    def forward_features(self, x):
        """Passes input through the backbone to get spatial features."""
        return self.backbone(x)

    def forward(self, image, contralateral):
        """
        Args:
            image (Tensor): Target breast image [B, C, H, W]
            contralateral (Tensor): Contralateral breast image [B, C, H, W]

        Returns:
            logits (Tensor): Raw prediction scores [B, 1]
        """
        # 1. Extract Features (Siamese Weights)
        # f_target, f_contra shape: [B, C, H', W']
        f_target = self.forward_features(image)
        f_contra = self.forward_features(contralateral)

        # 2. Spatial Difference
        # Subtract contralateral features from target features.
        # This mathematically cancels out symmetric global patterns (dense tissue)
        # while preserving local anomalies present only in the target.
        f_diff = f_target - f_contra

        # 3. Fusion
        # Concatenate:
        # - f_target: Provides texture/morphology info for characterization.
        # - f_diff: Provides explicit asymmetry signal for localization.
        f_fused = torch.cat([f_target, f_diff], dim=1)

        # 4. Head Processing
        x = self.head(f_fused)

        # 5. Classification
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.fc(x)

        return logits
