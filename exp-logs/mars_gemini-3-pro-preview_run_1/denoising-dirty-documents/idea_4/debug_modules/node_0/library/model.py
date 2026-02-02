import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


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

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Using bilinear upsampling followed by convolution as per task requirements
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # The input to DoubleConv will be the upsampled features + skip features
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Input is CHW. Pad x1 if it's smaller than x2 (skip connection)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map to output channels"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DeepSupervisionUNet(nn.Module):
    def __init__(self):
        super(DeepSupervisionUNet, self).__init__()

        n_channels = Config.IN_CHANNELS
        n_classes = Config.OUT_CHANNELS
        base = Config.BASE_FILTERS
        self.deep_supervision = Config.DEEP_SUPERVISION

        # Encoder
        self.inc = DoubleConv(n_channels, base)
        self.down1 = Down(base, base * 2)
        self.down2 = Down(base * 2, base * 4)
        self.down3 = Down(base * 4, base * 8)
        self.down4 = Down(base * 8, base * 16)

        # Decoder
        # up1: takes down4 (base*16) and down3 (base*8) -> outputs base*8
        self.up1 = Up(base * 16, base * 8, base * 8)

        # up2: takes up1 (base*8) and down2 (base*4) -> outputs base*4
        self.up2 = Up(base * 8, base * 4, base * 4)

        # up3: takes up2 (base*4) and down1 (base*2) -> outputs base*2
        self.up3 = Up(base * 4, base * 2, base * 2)

        # up4: takes up3 (base*2) and inc (base) -> outputs base
        self.up4 = Up(base * 2, base, base)

        # Final Output Head
        self.outc = OutConv(base, n_classes)

        # Deep Supervision Heads (Auxiliary Classifiers)
        if self.deep_supervision:
            # Attached to the output of intermediate upsampling blocks
            self.ds_out1 = OutConv(base * 8, n_classes)  # Output of up1
            self.ds_out2 = OutConv(base * 4, n_classes)  # Output of up2
            self.ds_out3 = OutConv(base * 2, n_classes)  # Output of up3

    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder Path
        d1 = self.up1(x5, x4)
        d2 = self.up2(d1, x3)
        d3 = self.up3(d2, x2)
        d4 = self.up4(d3, x1)

        # Final Prediction
        logits = self.outc(d4)

        if self.deep_supervision and self.training:
            # Return list of predictions: [final, aux_fine, aux_mid, aux_coarse]
            # The MultiScaleMSELoss will handle interpolation
            aux1 = self.ds_out1(d1)
            aux2 = self.ds_out2(d2)
            aux3 = self.ds_out3(d3)
            # Order: Final, then auxiliary heads (usually from closest to output to deepest)
            return [logits, aux3, aux2, aux1]
        else:
            return logits
