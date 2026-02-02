import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DoubleConv(nn.Module):
    """
    A helper module consisting of two rounds of:
    ReflectionPad2d -> Conv2d -> BatchNorm2d -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    """
    Standard 4-Level U-Net architecture with Reflection Padding and Sigmoid output.
    """

    def __init__(self):
        super(UNet, self).__init__()

        # Load configuration
        self.features = Config.ENCODER_FILTERS  # Expected: [32, 64, 128, 256, 512]
        in_channels = Config.CHANNELS

        # -------------------------------------------------------
        # Encoder (Downsampling Path)
        # -------------------------------------------------------
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Level 1: Input -> 32
        self.downs.append(DoubleConv(in_channels, self.features[0]))
        # Level 2: 32 -> 64
        self.downs.append(DoubleConv(self.features[0], self.features[1]))
        # Level 3: 64 -> 128
        self.downs.append(DoubleConv(self.features[1], self.features[2]))
        # Level 4: 128 -> 256
        self.downs.append(DoubleConv(self.features[2], self.features[3]))

        # -------------------------------------------------------
        # Bottleneck
        # -------------------------------------------------------
        # 256 -> 512
        self.bottleneck = DoubleConv(self.features[3], self.features[4])

        # -------------------------------------------------------
        # Decoder (Upsampling Path)
        # -------------------------------------------------------
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # We iterate to build the decoder layers:
        # Up 1: 512 -> 256 (Concat with Level 4)
        self.up_convs.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.features[4], self.features[3], kernel_size=3, padding=0),
            )
        )
        self.ups.append(
            DoubleConv(self.features[4], self.features[3])
        )  # In: 256+256=512, Out: 256

        # Up 2: 256 -> 128 (Concat with Level 3)
        self.up_convs.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.features[3], self.features[2], kernel_size=3, padding=0),
            )
        )
        self.ups.append(
            DoubleConv(self.features[3], self.features[2])
        )  # In: 128+128=256, Out: 128

        # Up 3: 128 -> 64 (Concat with Level 2)
        self.up_convs.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.features[2], self.features[1], kernel_size=3, padding=0),
            )
        )
        self.ups.append(
            DoubleConv(self.features[2], self.features[1])
        )  # In: 64+64=128, Out: 64

        # Up 4: 64 -> 32 (Concat with Level 1)
        self.up_convs.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.ReflectionPad2d(1),
                nn.Conv2d(self.features[1], self.features[0], kernel_size=3, padding=0),
            )
        )
        self.ups.append(
            DoubleConv(self.features[1], self.features[0])
        )  # In: 32+32=64, Out: 32

        # -------------------------------------------------------
        # Final Output
        # -------------------------------------------------------
        self.final_conv = nn.Conv2d(self.features[0], in_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck pass
        x = self.bottleneck(x)

        # Reverse skip connections for easy iteration in decoder
        skip_connections = skip_connections[::-1]

        # Decoder pass
        for i in range(len(self.ups)):
            x = self.up_convs[i](x)
            skip = skip_connections[i]

            # Ensure dimensions match for concatenation
            # (Robustness for input sizes not perfectly divisible by 16)
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )

            concat_skip = torch.cat((skip, x), dim=1)
            x = self.ups[i](concat_skip)

        # Final projection and bounded activation
        return torch.sigmoid(self.final_conv(x))
