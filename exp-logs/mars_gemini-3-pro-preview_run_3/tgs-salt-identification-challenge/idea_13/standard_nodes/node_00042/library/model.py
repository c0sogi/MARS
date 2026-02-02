import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) module.
    Ref: https://arxiv.org/abs/1803.02539
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    """
    Standard U-Net++ Decoder Block with scSE attention.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SCSEModule(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNeXt-50 (32x4d) encoder and scSE attention.
    Implements Deep Supervision with equal weighting.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # Encoder (ResNeXt-50)
        # ---------------------------------------------------------------------
        weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        self.encoder = resnext50_32x4d(weights=weights)

        # Encoder channel sizes for ResNeXt-50
        # x0_0: conv1 (Stride 2) -> 64
        # x1_0: layer1 (Stride 4) -> 256
        # x2_0: layer2 (Stride 8) -> 512
        # x3_0: layer3 (Stride 16) -> 1024
        # x4_0: layer4 (Stride 32) -> 2048
        c0, c1, c2, c3, c4 = 64, 256, 512, 1024, 2048

        # ---------------------------------------------------------------------
        # Decoder Configuration
        # ---------------------------------------------------------------------
        # Using the last 4 values from config for levels 3, 2, 1, 0
        # Config: [256, 128, 64, 32, 16] -> f3=128, f2=64, f1=32, f0=16
        decoder_channels = Config.DECODER_CHANNELS
        f3 = decoder_channels[1]
        f2 = decoder_channels[2]
        f1 = decoder_channels[3]
        f0 = decoder_channels[4]

        # ---------------------------------------------------------------------
        # U-Net++ Decoder Graph
        # ---------------------------------------------------------------------

        # --- Column j=1 ---
        # x3_1: Inputs x3_0, Up(x4_0)
        self.conv3_1 = DecoderBlock(c3 + c4, f3)

        # x2_1: Inputs x2_0, Up(x3_0)
        self.conv2_1 = DecoderBlock(c2 + c3, f2)

        # x1_1: Inputs x1_0, Up(x2_0)
        self.conv1_1 = DecoderBlock(c1 + c2, f1)

        # x0_1: Inputs x0_0, Up(x1_0)
        self.conv0_1 = DecoderBlock(c0 + c1, f0)

        # --- Column j=2 ---
        # x2_2: Inputs x2_0, x2_1, Up(x3_1)
        self.conv2_2 = DecoderBlock(c2 + f2 + f3, f2)

        # x1_2: Inputs x1_0, x1_1, Up(x2_1)
        self.conv1_2 = DecoderBlock(c1 + f1 + f2, f1)

        # x0_2: Inputs x0_0, x0_1, Up(x1_1)
        self.conv0_2 = DecoderBlock(c0 + f0 + f1, f0)

        # --- Column j=3 ---
        # x1_3: Inputs x1_0, x1_1, x1_2, Up(x2_2)
        self.conv1_3 = DecoderBlock(c1 + f1 + f1 + f2, f1)

        # x0_3: Inputs x0_0, x0_1, x0_2, Up(x1_2)
        self.conv0_3 = DecoderBlock(c0 + f0 + f0 + f1, f0)

        # --- Column j=4 ---
        # x0_4: Inputs x0_0, x0_1, x0_2, x0_3, Up(x1_3)
        self.conv0_4 = DecoderBlock(c0 + f0 + f0 + f0 + f1, f0)

        # ---------------------------------------------------------------------
        # Deep Supervision Heads
        # ---------------------------------------------------------------------
        self.final0_1 = nn.Conv2d(f0, 1, 1)
        self.final0_2 = nn.Conv2d(f0, 1, 1)
        self.final0_3 = nn.Conv2d(f0, 1, 1)
        self.final0_4 = nn.Conv2d(f0, 1, 1)

    def forward(self, x):
        input_size = x.shape[-2:]

        # ---------------------------------------------------------------------
        # Encoder Forward
        # ---------------------------------------------------------------------
        # x: [B, 3, H, W]

        # Layer 0 (Stem)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x0_0 = self.encoder.relu(x)  # Stride 2
        x = self.encoder.maxpool(x0_0)  # Stride 4

        # Layer 1-4
        x1_0 = self.encoder.layer1(x)  # Stride 4
        x2_0 = self.encoder.layer2(x1_0)  # Stride 8
        x3_0 = self.encoder.layer3(x2_0)  # Stride 16
        x4_0 = self.encoder.layer4(x3_0)  # Stride 32

        # ---------------------------------------------------------------------
        # Decoder Forward
        # ---------------------------------------------------------------------

        # --- Column j=1 ---
        # x3_1
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up(x4_0, x3_0)], dim=1))
        # x2_1
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up(x3_0, x2_0)], dim=1))
        # x1_1
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up(x2_0, x1_0)], dim=1))
        # x0_1
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up(x1_0, x0_0)], dim=1))

        # --- Column j=2 ---
        # x2_2
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up(x3_1, x2_0)], dim=1))
        # x1_2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up(x2_1, x1_0)], dim=1))
        # x0_2
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up(x1_1, x0_0)], dim=1))

        # --- Column j=3 ---
        # x1_3
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up(x2_2, x1_0)], dim=1))
        # x0_3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up(x1_2, x0_0)], dim=1))

        # --- Column j=4 ---
        # x0_4
        x0_4 = self.conv0_4(
            torch.cat([x0_0, x0_1, x0_2, x0_3, self._up(x1_3, x0_0)], dim=1)
        )

        # ---------------------------------------------------------------------
        # Heads & Output
        # ---------------------------------------------------------------------
        # All outputs are at Stride 2 (x0_j). Upsample to input resolution.

        out1 = self._final_up(self.final0_1(x0_1), input_size)
        out2 = self._final_up(self.final0_2(x0_2), input_size)
        out3 = self._final_up(self.final0_3(x0_3), input_size)
        out4 = self._final_up(self.final0_4(x0_4), input_size)

        if self.training:
            return [out1, out2, out3, out4]
        else:
            # During inference, return the highest fidelity output
            # Some strategies average them, but x0_4 is the most refined.
            return out4

    def _up(self, x, target):
        """Upsamples x to match target spatial dimensions."""
        if x.shape[-2:] != target.shape[-2:]:
            return F.interpolate(
                x, size=target.shape[-2:], mode="bilinear", align_corners=True
            )
        return x

    def _final_up(self, x, size):
        """Upsamples final logits to original image size."""
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)
