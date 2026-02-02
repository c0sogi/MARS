import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic Convolutional Block: Conv2d -> BatchNorm -> ReLU.
    Uses Reflection Padding to maintain statistical continuity at image boundaries.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block used in Encoder and Decoder levels.
    Consists of two consecutive ConvBlocks.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.net = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)


class BottleneckBlock(nn.Module):
    """
    Configurable Bottleneck Block.
    Supports variable depth to accommodate both the standard bottleneck (Stream A)
    and the Deep High-Capacity bottleneck (Stream B).
    """

    def __init__(self, in_channels, out_channels, depth):
        super(BottleneckBlock, self).__init__()
        layers = []

        # The first layer handles the transition from encoder output channels to bottleneck channels
        layers.append(ConvBlock(in_channels, out_channels))

        # Subsequent layers maintain the bottleneck channel dimension
        # We subtract 1 because the first layer is already added
        for _ in range(depth - 1):
            layers.append(ConvBlock(out_channels, out_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    """
    Upsampling Block implementing the 'bilinear_conv' strategy.
    Sequence: Bilinear Upsample -> Conv (reduce channels) -> Concat -> DoubleConv.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()

        # Bilinear upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution to reduce channels after upsampling (typically halving)
        # This keeps parameter count controlled before concatenation
        self.reduce_conv = nn.Conv2d(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
        )

        # Calculate input channels for the DoubleConv after concatenation
        # (in_channels // 2) comes from the upsampled path
        # skip_channels comes from the encoder skip connection
        concat_channels = (in_channels // 2) + skip_channels

        self.conv = DoubleConv(concat_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = self.reduce_conv(x)

        # Handle potential dimension mismatch due to odd input sizes
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Flexible U-Net Architecture for Heterogeneous Resolution-Capacity Ensemble.

    Can be instantiated as:
    1. Stream A (Context): depth=4, bottleneck_depth=2, large filters.
    2. Stream B (Diversity): depth=3, bottleneck_depth=8, smaller filters.
    """

    def __init__(
        self,
        depth,
        encoder_filters,
        bottleneck_filters,
        bottleneck_depth,
        in_channels=1,
        out_channels=1,
    ):
        super(UNet, self).__init__()

        self.depth = depth

        # --- Encoder Path ---
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_in_filters = in_channels
        for filters in encoder_filters:
            self.downs.append(DoubleConv(current_in_filters, filters))
            current_in_filters = filters

        # --- Bottleneck ---
        # Input to bottleneck is the output of the last encoder level
        self.bottleneck = BottleneckBlock(
            encoder_filters[-1], bottleneck_filters, bottleneck_depth
        )

        # --- Decoder Path ---
        self.ups = nn.ModuleList()
        current_in_filters = bottleneck_filters

        # Iterate backwards through encoder levels to build decoder
        # We match the output filters of each decoder block to the corresponding encoder level
        for i in range(depth - 1, -1, -1):
            skip_filters = encoder_filters[i]
            out_filters = skip_filters

            self.ups.append(UpBlock(current_in_filters, skip_filters, out_filters))
            current_in_filters = out_filters

        # --- Final Output ---
        # 1x1 Convolution to map to output channels (grayscale)
        self.final_conv = nn.Conv2d(encoder_filters[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        # Reverse skip connections for easy iteration
        skip_connections = skip_connections[::-1]

        for i, up in enumerate(self.ups):
            skip = skip_connections[i]
            x = up(x, skip)

        # Output
        x = self.final_conv(x)
        x = self.sigmoid(x)

        return x
