import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # "Upsampling: Bilinear Upsampling followed by Convolution"
        # We use bilinear upsampling to increase resolution, then a conv to reduce channels
        # to match the skip connection channels (out_channels), then concat, then DoubleConv.

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # Reduce channels by half after upsampling to prepare for concatenation
        self.conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        # The DoubleConv takes concatenated input (in_channels//2 + out_channels) -> out_channels
        # Note: in_channels passed to __init__ is the channel count coming FROM the deeper layer.
        # The skip connection will have 'out_channels' count.
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is from the deeper layer (needs upsampling)
        # x2 is the skip connection from the encoder
        x1 = self.up(x1)
        x1 = self.conv(x1)

        # Handle padding issues if input dimensions are not perfectly divisible by 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map to number of classes"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Flexible U-Net implementation supporting variable depth.

    Args:
        n_channels (int): Number of input channels (e.g., 1 for grayscale).
        n_classes (int): Number of output channels (e.g., 1 for grayscale regression).
        depth (int): Number of downsampling levels.
                     depth=4 corresponds to Stream A (Context).
                     depth=3 corresponds to Stream B (Texture).
        base_channels (int): Number of filters in the first layer.
    """

    def __init__(self, n_channels, n_classes, depth=4, base_channels=32):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth
        self.base_channels = base_channels

        # --- Encoder Construction ---
        self.inc = DoubleConv(n_channels, base_channels)

        self.downs = nn.ModuleList()
        current_channels = base_channels
        for _ in range(depth):
            # Downsample: C -> 2C
            self.downs.append(Down(current_channels, current_channels * 2))
            current_channels *= 2

        # --- Decoder Construction ---
        self.ups = nn.ModuleList()
        # We iterate backwards to construct Up layers corresponding to Downs
        for _ in range(depth):
            # Upsample: 2C -> C
            # Input to Up layer is 2C (from below), Output is C
            self.ups.append(Up(current_channels, current_channels // 2))
            current_channels //= 2

        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Store skip connections
        skips = []

        # Initial Conv
        x = self.inc(x)
        skips.append(x)

        # Encoder Path
        for i in range(self.depth):
            x = self.downs[i](x)
            # Store skip connection for all but the last down (which is the bottleneck)
            # Actually, standard UNet stores output of every level before maxpool.
            # Our Down block does MaxPool -> DoubleConv.
            # So the output of 'downs[i]' is the input to the next level.
            # We need to store the input to the Down block (which is the output of previous block).
            # Wait, let's adjust logic.
            # inc -> x (Level 0). Store x.
            # Down 0 (Level 0 -> Level 1). Output is Level 1. Store Level 1.
            # ...
            # The last output is the bottleneck. It doesn't need to be stored as a skip for itself.
            if i < self.depth - 1:
                skips.append(x)

        # At this point, 'x' is the bottleneck output.
        # 'skips' contains [Level 0, Level 1, ..., Level D-1]

        # Decoder Path
        # We iterate through ups. The first up layer corresponds to the last skip connection.
        for i in range(self.depth):
            # Pop the corresponding skip connection
            skip = skips.pop()
            x = self.ups[i](x, skip)

        logits = self.outc(x)
        return logits
