import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from library.config import Config


class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block: (Conv3x3 -> BN -> ReLU) x 2
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net Architecture with Bilinear Upsampling.
    Replaces Transposed Convolutions with Upsampling + Convolution to avoid checkerboard artifacts.
    Cite {solution_lesson_node_00009}
    """

    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        features=Config.FEATURES,
    ):
        super(UNet, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Encoder (Contracting Path) ---
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # --- Bottleneck ---
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # --- Decoder (Expanding Path) ---
        for feature in reversed(features):
            # Upsampling + Convolution
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=1),
                )
            )
            # Double Conv on concatenated features
            self.ups.append(DoubleConv(feature * 2, feature))

        # --- Final Output Layer ---
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Reverse skip connections
        skip_connections = skip_connections[::-1]

        # Decoder pass
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip_connection = skip_connections[i // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[i + 1](concat_skip)

        return self.final_conv(x)
