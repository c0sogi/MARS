import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel-wise attention.
    Recalibrates feature maps by explicitly modelling interdependencies between channels.
    """

    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        # Ensure reduction doesn't make hidden dim 0
        hidden_dim = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResidualBlock(nn.Module):
    """
    Residual Block with SiLU activation and Squeeze-and-Excitation.
    Standard building block for ResUNet.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.use_se = Config.USE_SE

        # Activation function configuration
        self.activation = (
            nn.SiLU(inplace=True) if Config.USE_SILU else nn.ReLU(inplace=True)
        )

        # First Convolution
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second Convolution
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Squeeze and Excitation
        if self.use_se:
            self.se = SqueezeExcitation(out_channels)

        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_se:
            out = self.se(out)

        out += residual
        out = self.activation(out)
        return out


class ResUNetPlusPlus(nn.Module):
    """
    Deeply Supervised ResUNet++ Architecture.
    Features:
    - Nested skip pathways (UNet++ topology).
    - Residual Blocks with SE and SiLU.
    - Transposed Convolution for upsampling.
    - Deep Supervision heads at level 0.
    """

    def __init__(self):
        super(ResUNetPlusPlus, self).__init__()

        # Filter configuration: [64, 128, 256, 512, 1024]
        nb_filter = [Config.BASE_FILTERS * (2**i) for i in range(5)]

        # Downsampling
        self.pool = nn.MaxPool2d(2, 2)

        # Helper for Transposed Convolution Upsampling
        def up_layer(in_ch, out_ch):
            return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)

        # ---------------------------------------------------------------------
        # Encoder (Backbone)
        # ---------------------------------------------------------------------
        # Input is 1 channel (grayscale) -> 64
        self.conv0_0 = ResidualBlock(1, nb_filter[0])
        self.conv1_0 = ResidualBlock(nb_filter[0], nb_filter[1])
        self.conv2_0 = ResidualBlock(nb_filter[1], nb_filter[2])
        self.conv3_0 = ResidualBlock(nb_filter[2], nb_filter[3])
        self.conv4_0 = ResidualBlock(nb_filter[3], nb_filter[4])

        # ---------------------------------------------------------------------
        # Nested Decoder Blocks (UNet++)
        # ---------------------------------------------------------------------

        # --- Level 0 (Original Resolution) ---
        # X_0_1: Inputs [X_0_0, Up(X_1_0)]
        self.up1_0_to_0_1 = up_layer(nb_filter[1], nb_filter[0])
        self.conv0_1 = ResidualBlock(nb_filter[0] * 2, nb_filter[0])

        # X_0_2: Inputs [X_0_0, X_0_1, Up(X_1_1)]
        self.up1_1_to_0_2 = up_layer(nb_filter[1], nb_filter[0])
        self.conv0_2 = ResidualBlock(nb_filter[0] * 3, nb_filter[0])

        # X_0_3: Inputs [X_0_0, X_0_1, X_0_2, Up(X_1_2)]
        self.up1_2_to_0_3 = up_layer(nb_filter[1], nb_filter[0])
        self.conv0_3 = ResidualBlock(nb_filter[0] * 4, nb_filter[0])

        # X_0_4: Inputs [X_0_0, X_0_1, X_0_2, X_0_3, Up(X_1_3)]
        self.up1_3_to_0_4 = up_layer(nb_filter[1], nb_filter[0])
        self.conv0_4 = ResidualBlock(nb_filter[0] * 5, nb_filter[0])

        # --- Level 1 ---
        # X_1_1: Inputs [X_1_0, Up(X_2_0)]
        self.up2_0_to_1_1 = up_layer(nb_filter[2], nb_filter[1])
        self.conv1_1 = ResidualBlock(nb_filter[1] * 2, nb_filter[1])

        # X_1_2: Inputs [X_1_0, X_1_1, Up(X_2_1)]
        self.up2_1_to_1_2 = up_layer(nb_filter[2], nb_filter[1])
        self.conv1_2 = ResidualBlock(nb_filter[1] * 3, nb_filter[1])

        # X_1_3: Inputs [X_1_0, X_1_1, X_1_2, Up(X_2_2)]
        self.up2_2_to_1_3 = up_layer(nb_filter[2], nb_filter[1])
        self.conv1_3 = ResidualBlock(nb_filter[1] * 4, nb_filter[1])

        # --- Level 2 ---
        # X_2_1: Inputs [X_2_0, Up(X_3_0)]
        self.up3_0_to_2_1 = up_layer(nb_filter[3], nb_filter[2])
        self.conv2_1 = ResidualBlock(nb_filter[2] * 2, nb_filter[2])

        # X_2_2: Inputs [X_2_0, X_2_1, Up(X_3_1)]
        self.up3_1_to_2_2 = up_layer(nb_filter[3], nb_filter[2])
        self.conv2_2 = ResidualBlock(nb_filter[2] * 3, nb_filter[2])

        # --- Level 3 ---
        # X_3_1: Inputs [X_3_0, Up(X_4_0)]
        self.up4_0_to_3_1 = up_layer(nb_filter[4], nb_filter[3])
        self.conv3_1 = ResidualBlock(nb_filter[3] * 2, nb_filter[3])

        # ---------------------------------------------------------------------
        # Deep Supervision Heads
        # ---------------------------------------------------------------------
        # 1x1 Convs to map feature maps to 1-channel noise residual
        self.final0_1 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final0_2 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final0_3 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)
        self.final0_4 = nn.Conv2d(nb_filter[0], 1, kernel_size=1)

    def forward(self, x):
        # -----------------------
        # Encoder Path
        # -----------------------
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        # -----------------------
        # Nested Decoder Path
        # -----------------------

        # Level 3
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0_to_3_1(x4_0)], 1))

        # Level 2
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0_to_2_1(x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up3_1_to_2_2(x3_1)], 1))

        # Level 1
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0_to_1_1(x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up2_1_to_1_2(x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up2_2_to_1_3(x2_2)], 1))

        # Level 0
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0_to_0_1(x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up1_1_to_0_2(x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up1_2_to_0_3(x1_2)], 1))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self.up1_3_to_0_4(x1_3)], 1)
        )

        # -----------------------
        # Deep Supervision Output
        # -----------------------
        out1 = self.final0_1(x0_1)
        out2 = self.final0_2(x0_2)
        out3 = self.final0_3(x0_3)
        out4 = self.final0_4(x0_4)

        if self.training and Config.DEEP_SUPERVISION:
            # Return list of outputs for multi-loss calculation
            return [out1, out2, out3, out4]
        else:
            # Inference: Return the output from the deepest, most refined branch
            return out4
