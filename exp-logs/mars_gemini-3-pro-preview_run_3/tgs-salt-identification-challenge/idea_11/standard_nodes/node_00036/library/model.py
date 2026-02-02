import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    """

    def __init__(self, channels, reduction=16):
        super(SCSEModule, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent: x * cSE(x) + x * sSE(x)
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard block for U-Net++ nodes: 2x (Conv3x3 -> BN -> ReLU) + scSE.
    """

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.scse(x)
        return x


class UpBlock(nn.Module):
    """
    Simple Bilinear Upsampling block.
    """

    def __init__(self):
        super(UpBlock, self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNeXt-50 Encoder and scSE Attention.
    """

    def __init__(
        self,
        encoder_name="resnext50_32x4d",
        in_channels=3,
        classes=1,
        deep_supervision=True,
    ):
        super(SaltUNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision

        # Load Encoder
        # features_only=True returns the feature maps at different strides
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, features_only=True, in_chans=in_channels
        )

        # Get encoder channel counts
        # Typically [64, 256, 512, 1024, 2048] for ResNeXt50
        enc_channels = self.encoder.feature_info.channels()

        # Define Decoder Channels for each level (0 to 4)
        # We define custom channel widths for the decoder nodes to control complexity
        self.filters = [32, 64, 128, 256, 512]

        # --- Decoder Nodes Construction ---
        # Notation: conv{level}_{depth}

        # Level 4 (Stride 32)
        # Only x4_0 exists (Encoder output)

        # Level 3 (Stride 16)
        # x3_1 inputs: x3_0 (Enc), Up(x4_0)
        self.conv3_1 = ConvBlock(enc_channels[3] + enc_channels[4], self.filters[3])

        # Level 2 (Stride 8)
        # x2_1 inputs: x2_0 (Enc), Up(x3_0)
        self.conv2_1 = ConvBlock(enc_channels[2] + enc_channels[3], self.filters[2])
        # x2_2 inputs: x2_0, x2_1, Up(x3_1)
        self.conv2_2 = ConvBlock(
            enc_channels[2] + self.filters[2] + self.filters[3], self.filters[2]
        )

        # Level 1 (Stride 4)
        # x1_1 inputs: x1_0 (Enc), Up(x2_0)
        self.conv1_1 = ConvBlock(enc_channels[1] + enc_channels[2], self.filters[1])
        # x1_2 inputs: x1_0, x1_1, Up(x2_1)
        self.conv1_2 = ConvBlock(
            enc_channels[1] + self.filters[1] + self.filters[2], self.filters[1]
        )
        # x1_3 inputs: x1_0, x1_1, x1_2, Up(x2_2)
        self.conv1_3 = ConvBlock(
            enc_channels[1] + 2 * self.filters[1] + self.filters[2], self.filters[1]
        )

        # Level 0 (Stride 2)
        # x0_1 inputs: x0_0 (Enc), Up(x1_0)
        self.conv0_1 = ConvBlock(enc_channels[0] + enc_channels[1], self.filters[0])
        # x0_2 inputs: x0_0, x0_1, Up(x1_1)
        self.conv0_2 = ConvBlock(
            enc_channels[0] + self.filters[0] + self.filters[1], self.filters[0]
        )
        # x0_3 inputs: x0_0, x0_1, x0_2, Up(x1_2)
        self.conv0_3 = ConvBlock(
            enc_channels[0] + 2 * self.filters[0] + self.filters[1], self.filters[0]
        )
        # x0_4 inputs: x0_0, x0_1, x0_2, x0_3, Up(x1_3)
        self.conv0_4 = ConvBlock(
            enc_channels[0] + 3 * self.filters[0] + self.filters[1], self.filters[0]
        )

        # Upsampling Helper
        self.up = UpBlock()

        # Final Output Heads (Deep Supervision)
        # The decoder outputs (x0_j) are at Stride 2. We need to upsample to Stride 1 (Input Resolution).
        self.final0_1 = nn.Sequential(UpBlock(), nn.Conv2d(self.filters[0], classes, 1))
        self.final0_2 = nn.Sequential(UpBlock(), nn.Conv2d(self.filters[0], classes, 1))
        self.final0_3 = nn.Sequential(UpBlock(), nn.Conv2d(self.filters[0], classes, 1))
        self.final0_4 = nn.Sequential(UpBlock(), nn.Conv2d(self.filters[0], classes, 1))

    def forward(self, x):
        # x: (B, 3, H, W)

        # --- Encoder ---
        features = self.encoder(x)
        x0_0 = features[0]  # Stride 2
        x1_0 = features[1]  # Stride 4
        x2_0 = features[2]  # Stride 8
        x3_0 = features[3]  # Stride 16
        x4_0 = features[4]  # Stride 32

        # --- Decoder ---

        # Level 3
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))

        # Level 2
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        # Level 1
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))

        # Level 0
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        # --- Output Heads ---
        out1 = self.final0_1(x0_1)
        out2 = self.final0_2(x0_2)
        out3 = self.final0_3(x0_3)
        out4 = self.final0_4(x0_4)

        # Return list for deep supervision training, single tensor for inference
        if self.deep_supervision and self.training:
            return [out1, out2, out3, out4]
        else:
            return out4
