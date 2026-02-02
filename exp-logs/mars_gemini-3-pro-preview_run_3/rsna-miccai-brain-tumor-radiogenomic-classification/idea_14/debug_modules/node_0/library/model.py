import torch
import torch.nn as nn
import timm
from library.config import Config


class StabilizedStem(nn.Module):
    """
    A specialized stem for the 2.5D network that processes high-density volumetric input.

    Structure:
    1. Intra-Slice Multi-Modal Learning: Grouped Convolution (groups=32)
       - Input: 128 channels (32 slices * 4 modalities)
       - Groups: 32 (one group per slice)
       - Each group processes 4 channels (FLAIR, T1w, T1wCE, T2w for that slice)
    2. Depth Compression: Pointwise Convolution
       - Compresses 128 channels -> 64 channels to match backbone input.

    Includes explicit Kaiming/He Normal initialization for stability.
    """

    def __init__(self, in_channels=128, mid_channels=128, out_channels=64, groups=32):
        super(StabilizedStem, self).__init__()

        # Layer 1: Intra-Slice Multi-Modal Learning
        # Groups=32 ensures each kernel sees exactly 4 channels (1 slice's modalities)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.ReLU(inplace=True)

        # Layer 2: Depth Compression
        # Mixes features across the Z-axis (slices) and projects to backbone input size
        self.conv2 = nn.Conv2d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU(inplace=True)

        self._init_weights()

    def _init_weights(self):
        """
        Explicitly applies Kaiming/He Normal initialization to convolutional layers
        to prevent gradient explosion/vanishing with high-channel inputs.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x


class BraTS25DNet(nn.Module):
    """
    The main 2.5D Convolutional Neural Network.

    Components:
    1. StabilizedStem: Processes (B, 128, H, W) input.
    2. Backbone: EfficientNet-B0 (in_chans=64).
    3. Head: Global Average Pooling + Linear (handled by timm).
    """

    def __init__(self):
        super(BraTS25DNet, self).__init__()

        # Configuration from prompt/config
        # Input: 32 slices * 4 modalities = 128 channels
        in_channels = Config.IN_CHANNELS
        stem_out_channels = Config.STEM_OUT_CHANNELS

        # 1. Stabilized Slice-Grouped Stem
        self.stem = StabilizedStem(
            in_channels=in_channels,
            mid_channels=in_channels,  # Keep dim same in first layer
            out_channels=stem_out_channels,
            groups=Config.NUM_SLICES,  # 32 groups
        )

        # 2. Backbone & Head
        # We use timm to create EfficientNet-B0.
        # in_chans=64 matches the stem output.
        # num_classes=1 creates the linear head for binary classification.
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=stem_out_channels,
            num_classes=1,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, 128, 256, 256)
        Returns:
            torch.Tensor: Logits of shape (B, 1)
        """
        # Pass through stem
        x = self.stem(x)

        # Pass through backbone (includes GAP and Classifier Head)
        x = self.backbone(x)

        return x
