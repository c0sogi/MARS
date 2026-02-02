import torch
import torch.nn as nn
import timm
from library.config import IN_CHANNELS


class SliceGroupedFusionNet(nn.Module):
    """
    2.5D Convolutional Neural Network with a Slice-Grouped Stem.

    Architecture:
    1. Input: (B, 128, 256, 256) - 32 slices * 4 modalities
    2. Slice-Grouped Stem:
       - Conv2d(groups=32): Enforces intra-slice multi-modal learning.
       - Conv2d(1x1): Aggregates features across the depth axis.
    3. Backbone: EfficientNet-B0 (in_chans=64)
    4. Head: Global Average Pooling -> Linear -> Logits
    """

    def __init__(
        self, in_channels=IN_CHANNELS, backbone_name="efficientnet_b0", pretrained=True
    ):
        super(SliceGroupedFusionNet, self).__init__()

        # ==========================================
        # Slice-Grouped Stem
        # ==========================================
        # Layer 1: Intra-Slice Multi-Modal Learning
        # Input: 128 channels (32 slices * 4 modalities)
        # Groups: 32. Each group sees 4 channels (1 slice's modalities).
        # Output: 64 channels. Each group outputs 2 channels.
        self.stem_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,
            groups=32,
            bias=False,
        )
        self.stem_bn1 = nn.BatchNorm2d(64)
        self.stem_act1 = nn.ReLU(inplace=True)

        # Layer 2: Inter-Slice Aggregation
        # Mixes the features learned from different slices/groups
        self.stem_conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=1, bias=False
        )
        self.stem_bn2 = nn.BatchNorm2d(64)
        self.stem_act2 = nn.ReLU(inplace=True)

        # Apply Kaiming Normal Initialization to Stem
        self._init_stem_weights()

        # ==========================================
        # Backbone (EfficientNet-B0)
        # ==========================================
        # in_chans=64 to match stem output
        # num_classes=0 returns the pooled feature vector
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, in_chans=64, num_classes=0
        )

        # Get feature dimension (EfficientNet-B0 usually 1280)
        self.num_features = self.backbone.num_features

        # ==========================================
        # Head
        # ==========================================
        # Global Average Pooling is handled by the backbone when num_classes=0
        self.fc = nn.Linear(self.num_features, 1)

    def _init_stem_weights(self):
        """
        Initializes stem layers with Kaiming Normal to prevent gradient issues
        with high-dimensional inputs.
        """
        for m in [self.stem_conv1, self.stem_conv2]:
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        for m in [self.stem_bn1, self.stem_bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, 128, 256, 256)

        # Stem
        x = self.stem_conv1(x)
        x = self.stem_bn1(x)
        x = self.stem_act1(x)

        x = self.stem_conv2(x)
        x = self.stem_bn2(x)
        x = self.stem_act2(x)

        # Backbone (returns pooled features)
        # Shape: (B, 1280)
        x = self.backbone(x)

        # Head
        # Shape: (B, 1)
        logits = self.fc(x)

        return logits
