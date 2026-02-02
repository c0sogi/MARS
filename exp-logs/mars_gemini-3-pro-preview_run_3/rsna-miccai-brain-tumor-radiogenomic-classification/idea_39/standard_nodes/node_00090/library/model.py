import torch
import torch.nn as nn
import timm
from library.config import Config


class SSFNet(nn.Module):
    """
    Siamese Spatially-Fused 2.5D Network (SSF-Net).

    Architecture:
    1. Shared Backbone (EfficientNet-B0): Processes Even and Odd slice streams independently.
       - Input: (B, 64, 224, 224)
       - Output: Spatial Features (B, 1280, 7, 7)
    2. Spatial Fusion:
       - Concatenates features: (B, 2560, 7, 7)
       - Compresses via 1x1 Conv: (B, 1280, 7, 7)
    3. Head:
       - Global Average Pooling
       - Linear Classifier -> Logits
    """

    def __init__(self):
        super(SSFNet, self).__init__()

        # 1. Shared Backbone
        # We rely on timm's weight recycling to adapt the first layer to 64 channels.
        # global_pool='' and num_classes=0 ensure we get spatial feature maps.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            in_chans=Config.IN_CHANS,
            num_classes=0,
            global_pool="",
            drop_path_rate=Config.DROP_PATH_RATE,
        )

        # Dynamically determine feature dimension (expected 1280 for EfficientNet-B0)
        with torch.no_grad():
            dummy_input = torch.randn(
                1, Config.IN_CHANS, Config.IMG_SIZE, Config.IMG_SIZE
            )
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # 2. Fusion Block
        # Processes the concatenated features from both streams.
        # Input channels: num_features * 2 (Even + Odd)
        # Output channels: num_features
        self.fusion = nn.Sequential(
            nn.Conv2d(
                self.num_features * 2, self.num_features, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.num_features),
            nn.ReLU(inplace=True),
        )

        # 3. Classification Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x_even, x_odd):
        """
        Args:
            x_even: Tensor of shape (B, 64, 224, 224)
            x_odd:  Tensor of shape (B, 64, 224, 224)
        Returns:
            logits: Tensor of shape (B, 1)
        """
        # Pass both streams through the shared backbone
        f_even = self.backbone(x_even)  # (B, C, H, W)
        f_odd = self.backbone(x_odd)  # (B, C, H, W)

        # Concatenate along channel dimension
        f_cat = torch.cat([f_even, f_odd], dim=1)  # (B, 2*C, H, W)

        # Apply Spatial Fusion
        f_fused = self.fusion(f_cat)  # (B, C, H, W)

        # Pooling and Classification
        f_pool = self.global_pool(f_fused)  # (B, C, 1, 1)
        f_flat = f_pool.flatten(1)  # (B, C)
        logits = self.classifier(f_flat)  # (B, 1)

        return logits
