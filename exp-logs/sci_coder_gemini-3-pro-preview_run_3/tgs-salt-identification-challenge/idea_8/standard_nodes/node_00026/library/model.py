import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Module.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolution Block with SCSE attention.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> SCSE
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x


class SaltModel(nn.Module):
    """
    U-Net++ with ResNeXt-50 (32x4d) Encoder and Deep Supervision.
    """

    def __init__(self, encoder_name="resnext50_32x4d", pretrained=True, in_channels=3):
        super().__init__()

        # 1. Encoder (ResNeXt-50)
        # features_only=True returns a list of features from the backbone
        # indices=(0, 1, 2, 3, 4) corresponds to strides (2, 4, 8, 16, 32)
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Get channel counts dynamically
        # For ResNeXt50: [64, 256, 512, 1024, 2048]
        enc_channels = self.encoder.feature_info.channels()

        # Decoder target channels for each level (L0 to L3)
        # We use a standard reduction: 256 -> 128 -> 64 -> 32
        # L3 (H/16), L2 (H/8), L1 (H/4), L0 (H/2)
        dec_channels = [32, 64, 128, 256]

        # ---------------------------------------------------------------------
        # Decoder Blocks (U-Net++ Nested Structure)
        # Notation: conv{level}_{nesting_index}
        # ---------------------------------------------------------------------

        # Level 3 (H/16)
        # Input: Encoder L3 + Upsampled Encoder L4
        self.conv3_1 = ConvBlock(enc_channels[3] + enc_channels[4], dec_channels[3])

        # Level 2 (H/8)
        # Input: Encoder L2 + Upsampled Encoder L3
        self.conv2_1 = ConvBlock(enc_channels[2] + enc_channels[3], dec_channels[2])
        # Input: Encoder L2 + Node 2_1 + Upsampled Node 3_1
        self.conv2_2 = ConvBlock(
            enc_channels[2] + dec_channels[2] + dec_channels[3], dec_channels[2]
        )

        # Level 1 (H/4)
        # Input: Encoder L1 + Upsampled Encoder L2
        self.conv1_1 = ConvBlock(enc_channels[1] + enc_channels[2], dec_channels[1])
        # Input: Encoder L1 + Node 1_1 + Upsampled Node 2_1
        self.conv1_2 = ConvBlock(
            enc_channels[1] + dec_channels[1] + dec_channels[2], dec_channels[1]
        )
        # Input: Encoder L1 + Node 1_1 + Node 1_2 + Upsampled Node 2_2
        self.conv1_3 = ConvBlock(
            enc_channels[1] + dec_channels[1] * 2 + dec_channels[2], dec_channels[1]
        )

        # Level 0 (H/2) - Note: ResNet stem starts at stride 2
        # Input: Encoder L0 + Upsampled Encoder L1
        self.conv0_1 = ConvBlock(enc_channels[0] + enc_channels[1], dec_channels[0])
        # Input: Encoder L0 + Node 0_1 + Upsampled Node 1_1
        self.conv0_2 = ConvBlock(
            enc_channels[0] + dec_channels[0] + dec_channels[1], dec_channels[0]
        )
        # Input: Encoder L0 + Node 0_1 + Node 0_2 + Upsampled Node 1_2
        self.conv0_3 = ConvBlock(
            enc_channels[0] + dec_channels[0] * 2 + dec_channels[1], dec_channels[0]
        )
        # Input: Encoder L0 + Node 0_1 + Node 0_2 + Node 0_3 + Upsampled Node 1_3
        self.conv0_4 = ConvBlock(
            enc_channels[0] + dec_channels[0] * 3 + dec_channels[1], dec_channels[0]
        )

        # ---------------------------------------------------------------------
        # Deep Supervision Heads
        # ---------------------------------------------------------------------
        self.final0_1 = nn.Conv2d(dec_channels[0], 1, kernel_size=1)
        self.final0_2 = nn.Conv2d(dec_channels[0], 1, kernel_size=1)
        self.final0_3 = nn.Conv2d(dec_channels[0], 1, kernel_size=1)
        self.final0_4 = nn.Conv2d(dec_channels[0], 1, kernel_size=1)

    def _up(self, x, target):
        """Upsamples x to match target's spatial dimensions."""
        if x.shape[2:] != target.shape[2:]:
            return F.interpolate(
                x, size=target.shape[2:], mode="bilinear", align_corners=True
            )
        return x

    def _final_up(self, x, size):
        """Upsamples x to the final output size."""
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

    def forward(self, x):
        input_size = x.shape[2:]

        # 1. Encoder Pass
        # Returns [x0_0, x1_0, x2_0, x3_0, x4_0]
        # Strides: [2, 4, 8, 16, 32]
        enc_feats = self.encoder(x)
        x0_0, x1_0, x2_0, x3_0, x4_0 = enc_feats

        # 2. Decoder Pass (Nested)

        # Level 3
        # x3_1 = Block(x3_0, Up(x4_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], 1))

        # Level 2
        # x2_1 = Block(x2_0, Up(x3_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], 1))
        # x2_2 = Block(x2_0, x2_1, Up(x3_1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], 1))

        # Level 1
        # x1_1 = Block(x1_0, Up(x2_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], 1))
        # x1_2 = Block(x1_0, x1_1, Up(x2_1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], 1))
        # x1_3 = Block(x1_0, x1_1, x1_2, Up(x2_2))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], 1))

        # Level 0
        # x0_1 = Block(x0_0, Up(x1_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], 1))
        # x0_2 = Block(x0_0, x0_1, Up(x1_1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], 1))
        # x0_3 = Block(x0_0, x0_1, x0_2, Up(x1_2))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], 1))
        # x0_4 = Block(x0_0, x0_1, x0_2, x0_3, Up(x1_3))
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], 1)
        )

        # 3. Deep Supervision Heads
        # All outputs are upsampled to original input size
        logits_0_1 = self._final_up(self.final0_1(x0_1), input_size)
        logits_0_2 = self._final_up(self.final0_2(x0_2), input_size)
        logits_0_3 = self._final_up(self.final0_3(x0_3), input_size)
        logits_0_4 = self._final_up(self.final0_4(x0_4), input_size)

        if self.training:
            return [logits_0_1, logits_0_2, logits_0_3, logits_0_4]
        else:
            return logits_0_4
