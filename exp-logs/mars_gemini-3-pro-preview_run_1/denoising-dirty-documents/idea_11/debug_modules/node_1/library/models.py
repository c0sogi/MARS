import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import CONTEXT_MODEL_CONFIG, DIVERSITY_MODEL_CONFIG


class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    """

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
    """
    Downscaling with maxpool then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling followed by a convolution to reduce channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # The input to DoubleConv will be (in_channels + skip_channels)
            # We assume in_channels comes from the deeper layer (which we upsample)
            # and we concat with skip_channels (which is usually out_channels size)
            self.conv = DoubleConv(
                in_channels + out_channels, out_channels, out_channels
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Handle padding if dimensions don't match exactly due to odd sizes
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth, filters, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.bilinear = bilinear

        # Verify filters length matches depth + 1 (input_block + depth * down_blocks is not quite right)
        # Structure:
        # Inc (Input -> filters[0])
        # Down 1..depth (filters[i] -> filters[i+1])
        # Up 1..depth
        if len(filters) != depth + 1:
            raise ValueError(
                f"Filter list length {len(filters)} must be depth {depth} + 1"
            )

        self.inc = DoubleConv(in_channels, filters[0])

        self.downs = nn.ModuleList()
        for i in range(depth):
            self.downs.append(Down(filters[i], filters[i + 1]))

        self.ups = nn.ModuleList()
        # Iterate backwards for upsampling:
        # Bottleneck is at filters[depth]
        # First up takes filters[depth] and concats with filters[depth-1] -> outputs filters[depth-1]
        for i in range(depth, 0, -1):
            self.ups.append(Up(filters[i], filters[i - 1], bilinear))

        self.outc = OutConv(filters[0], out_channels)

    def forward(self, x):
        x_skips = []

        # Initial Conv
        x = self.inc(x)
        x_skips.append(x)

        # Downsampling path
        for down in self.downs:
            x = down(x)
            x_skips.append(x)

        # The last element in x_skips is the bottleneck output, which we don't need to skip to itself
        # But for the loop logic, we pop from the list.
        # Actually, standard UNet:
        # Inc -> x1
        # Down1(x1) -> x2
        # ...
        # DownLast -> Bottleneck
        # Up1(Bottleneck, xLast)

        # Current logic:
        # x_skips[0] is output of Inc (filters[0])
        # x_skips[1] is output of Down1 (filters[1])
        # ...
        # x_skips[depth] is output of DownDepth (filters[depth]) -> This is the bottleneck

        x = x_skips.pop()  # Bottleneck

        # Upsampling path
        for up in self.ups:
            skip = x_skips.pop()
            x = up(x, skip)

        logits = self.outc(x)
        return torch.sigmoid(logits)


def build_context_model():
    """
    Builds the Stream A (Context Specialist) model.
    4-Level U-Net, Input 320x320.
    """
    config = CONTEXT_MODEL_CONFIG
    return UNet(
        in_channels=1,
        out_channels=1,
        depth=config["unet_depth"],
        filters=config["encoder_filters"],
        bilinear=True,
    )


def build_diversity_model():
    """
    Builds the Stream B (Diversity Specialist) model.
    3-Level U-Net, Input 160x160.
    """
    config = DIVERSITY_MODEL_CONFIG
    return UNet(
        in_channels=1,
        out_channels=1,
        depth=config["unet_depth"],
        filters=config["encoder_filters"],
        bilinear=True,
    )
