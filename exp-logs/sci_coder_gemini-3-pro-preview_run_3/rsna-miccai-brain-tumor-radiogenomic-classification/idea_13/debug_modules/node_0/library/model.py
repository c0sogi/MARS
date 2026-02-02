import torch
import torch.nn as nn
import timm
from library import config


class SliceGroupedStem(nn.Module):
    """
    Attention-Weighted Slice-Grouped Stem.

    Processing Steps:
    1. Grouped Convolution (groups=32): Processes each slice's 4 modalities independently
       to extract intra-slice features.
    2. Slice-Wise Attention (SE Block): Learns a weighting vector for the 32 slices
       to suppress noise (e.g., empty skull slices) and highlight tumor regions.
    3. Channel Compression: Reduces the feature depth to match the backbone's input requirements.
    """

    def __init__(
        self, in_channels=128, stem_groups=32, mid_channels=128, out_channels=64
    ):
        super(SliceGroupedStem, self).__init__()

        # Layer 1: Intra-Slice Feature Extraction
        # groups=32 ensures each group sees exactly 4 channels (the 4 modalities of one slice)
        self.conv1 = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            groups=stem_groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)

        # Layer 2: Slice-Wise Attention (SE Block)
        # Global Average Pooling -> FC -> ReLU -> FC -> Sigmoid
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Reduction ratio of 4 (128 -> 32 hidden channels)
        reduction_channels = mid_channels // 4

        self.se_block = nn.Sequential(
            nn.Conv2d(mid_channels, reduction_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, mid_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Layer 3: Channel Compression
        # Compresses 128 channels down to 64
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        # Input shape: (Batch, 128, H, W)

        # 1. Grouped Conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        # 2. Attention
        scale = self.global_pool(x)
        scale = self.se_block(scale)
        x = x * scale

        # 3. Compression
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)

        return x


class MGMTNet(nn.Module):
    """
    2.5D Convolutional Neural Network for MGMT Promoter Methylation Prediction.

    Structure:
    - Input: (B, 128, 256, 256) Interleaved Slices
    - Stem: SliceGroupedStem (Custom)
    - Backbone: EfficientNet-B0 (timm)
    - Head: Global Pooling + Linear (part of backbone)
    """

    def __init__(self):
        super(MGMTNet, self).__init__()

        # 1. Custom Stem
        self.stem = SliceGroupedStem(
            in_channels=config.INPUT_CHANNELS,  # 128
            stem_groups=config.STEM_GROUPS,  # 32
            mid_channels=config.STEM_OUT_CHANNELS,  # 128
            out_channels=config.COMPRESSED_CHANNELS,  # 64
        )

        # 2. Backbone
        # Load EfficientNet-B0, modify first layer to accept 64 channels
        self.backbone = timm.create_model(
            config.BACKBONE_NAME,
            pretrained=True,
            in_chans=config.COMPRESSED_CHANNELS,
            num_classes=1,
        )

        # 3. Initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize Stem using Kaiming Normal (He Init)
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Backbone is pretrained, so we leave it.
        # The classification head created by timm is initialized by timm.

    def forward(self, x):
        # x: (B, 128, 256, 256)

        # Pass through Stem
        x = self.stem(x)  # -> (B, 64, 256, 256)

        # Pass through Backbone + Head
        # Returns logits (B, 1)
        logits = self.backbone(x)

        return logits
