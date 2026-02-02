import torch
import torch.nn as nn
import torch.nn.functional as F
from library import config


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
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

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, mid_channels=in_channels // 2
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Handle padding if dimensions don't match exactly due to odd sizes
        # (Though with 160x160 and 3 levels, sizes should be exact: 160->80->40->20)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class WideBottleneckUNet(nn.Module):
    def __init__(self, n_channels=config.IN_CHANNELS, n_classes=config.OUT_CHANNELS):
        super(WideBottleneckUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # --- Encoder (3 Levels) ---
        # Level 1: Input -> 32
        self.inc = DoubleConv(n_channels, 32)
        # Level 2: 32 -> 64
        self.down1 = Down(32, 64)
        # Level 3: 64 -> 128
        self.down2 = Down(64, 128)

        # --- Deep Wide Bottleneck ---
        # Input comes from Level 3 (128 channels) after a MaxPool
        self.bottleneck_pool = nn.MaxPool2d(2)

        # Sequence: 128->256->256->512->512->256->256
        self.bottleneck_conv = nn.Sequential(
            # 128 -> 256
            nn.Conv2d(
                128, 256, kernel_size=3, padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 -> 256
            nn.Conv2d(
                256, 256, kernel_size=3, padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 -> 512
            nn.Conv2d(
                256, 512, kernel_size=3, padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 -> 512
            nn.Conv2d(
                512, 512, kernel_size=3, padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 512 -> 256
            nn.Conv2d(
                512, 256, kernel_size=3, padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 256 -> 256
            nn.Conv2d(
                256, 256, kernel_size=3, padding=1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # --- Decoder ---
        # Up 1: Input 256 (Bottleneck) + 128 (Skip Level 3) = 384 -> Output 128
        self.up1 = Up(384, 128)

        # Up 2: Input 128 (Up1) + 64 (Skip Level 2) = 192 -> Output 64
        self.up2 = Up(192, 64)

        # Up 3: Input 64 (Up2) + 32 (Skip Level 1) = 96 -> Output 32
        self.up3 = Up(96, 32)

        # --- Output ---
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)  # (B, 32, H, W)
        x2 = self.down1(x1)  # (B, 64, H/2, W/2)
        x3 = self.down2(x2)  # (B, 128, H/4, W/4)

        # Bottleneck
        x_pool = self.bottleneck_pool(x3)  # (B, 128, H/8, W/8)
        x_bot = self.bottleneck_conv(x_pool)  # (B, 256, H/8, W/8)

        # Decoder
        x = self.up1(x_bot, x3)  # (B, 128, H/4, W/4)
        x = self.up2(x, x2)  # (B, 64, H/2, W/2)
        x = self.up3(x, x1)  # (B, 32, H, W)

        logits = self.outc(x)
        return self.sigmoid(logits)
