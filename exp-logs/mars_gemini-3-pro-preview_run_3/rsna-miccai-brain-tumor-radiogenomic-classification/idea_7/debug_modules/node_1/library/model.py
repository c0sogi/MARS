import torch
import torch.nn as nn
import timm
from library import config


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel-wise attention.
    Performs global average pooling, followed by a reduction/expansion FC network
    to rescale input channels based on their importance.
    """

    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling -> (B, C)
        y = self.avg_pool(x).view(b, c)
        # Excitation: FC Layers -> (B, C, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: Reweight channels
        return x * y


class MGMTNet(nn.Module):
    """
    Channel-Attention Compressed 2.5D Network.

    Architecture:
    1. Input: (B, 128, 256, 256) - 128 channels from 4 modalities * 32 slices.
    2. Stem:
       - SEBlock: Learns to suppress empty/irrelevant slices.
       - Bottleneck: 1x1 Conv reduces 128 channels -> 64 channels.
    3. Backbone: EfficientNet-B0 (modified input) -> Global Average Pooling -> Linear.
    4. Output: Single logit for binary classification.
    """

    def __init__(self):
        super(MGMTNet, self).__init__()

        # Retrieve hyperparameters from config
        in_channels = config.IN_CHANNELS
        reduction = config.STEM_REDUCTION_RATIO
        bottleneck_channels = config.BOTTLENECK_CHANNELS
        backbone_name = config.BACKBONE_NAME

        # 1. Channel-Attention Stem
        self.se_block = SEBlock(in_channels, reduction)

        # 2. Channel Bottleneck
        # Linear compression from 128 to 64 channels using 1x1 convolution.
        # Bias is False as it typically feeds into a normalization layer (in the backbone).
        self.bottleneck = nn.Conv2d(
            in_channels=in_channels,
            out_channels=bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # 3. Backbone
        # EfficientNet-B0 initialized with pretrained weights.
        # Modified to accept 'bottleneck_channels' (64) as input.
        # Output is set to 1 class (logit).
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, in_chans=bottleneck_channels, num_classes=1
        )

    def forward(self, x):
        # x shape: (B, 128, 256, 256)

        # Apply Channel Attention (Soft Slice Selection)
        x = self.se_block(x)

        # Apply Channel Compression
        x = self.bottleneck(x)

        # Apply Backbone and Classification Head
        # Output shape: (B, 1)
        x = self.backbone(x)

        return x
