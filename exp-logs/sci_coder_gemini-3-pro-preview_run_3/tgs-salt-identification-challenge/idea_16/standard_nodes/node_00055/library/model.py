import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    Enhances meaningful features by recalibrating channel and spatial information.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation (cSE)
        # Global Average Pooling -> MLP -> Sigmoid -> Scale channels
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation (sSE)
        # Conv1x1 -> Sigmoid -> Scale spatial map
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent combination (Add)
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolution Block for U-Net++ nodes.
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> scSE
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


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ Architecture with SE-ResNeXt-50 Encoder and scSE Attention.
    Supports Dynamic Deep Supervision for two-phase training.
    """

    def __init__(self, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        # 1. Encoder (SE-ResNeXt-50 32x4d)
        # features_only=True returns features at strides [2, 4, 8, 16, 32]
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Encoder Channel Depths
        # e0 (stem): 64, e1: 256, e2: 512, e3: 1024, e4: 2048
        e_ch = [64, 256, 512, 1024, 2048]

        # Decoder Channel Depths (Lightweight capacity)
        # d0: 16, d1: 32, d2: 64, d3: 128
        d_ch = [16, 32, 64, 128]

        # 2. Nested Decoder Blocks

        # Level 3 (Input from e3, e4)
        # X^{3,1} inputs: X^{3,0} (e3) + Up(X^{4,0} (e4))
        self.conv3_1 = ConvBlock(e_ch[3] + e_ch[4], d_ch[3])

        # Level 2
        # X^{2,1} inputs: X^{2,0} (e2) + Up(X^{3,0} (e3))
        self.conv2_1 = ConvBlock(e_ch[2] + e_ch[3], d_ch[2])
        # X^{2,2} inputs: X^{2,0} (e2) + X^{2,1} + Up(X^{3,1})
        self.conv2_2 = ConvBlock(e_ch[2] + d_ch[2] + d_ch[3], d_ch[2])

        # Level 1
        # X^{1,1} inputs: X^{1,0} (e1) + Up(X^{2,0} (e2))
        self.conv1_1 = ConvBlock(e_ch[1] + e_ch[2], d_ch[1])
        # X^{1,2} inputs: X^{1,0} (e1) + X^{1,1} + Up(X^{2,1})
        self.conv1_2 = ConvBlock(e_ch[1] + d_ch[1] + d_ch[2], d_ch[1])
        # X^{1,3} inputs: X^{1,0} (e1) + X^{1,1} + X^{1,2} + Up(X^{2,2})
        self.conv1_3 = ConvBlock(e_ch[1] + d_ch[1] + d_ch[1] + d_ch[2], d_ch[1])

        # Level 0 (Output Stride 2)
        # X^{0,1} inputs: X^{0,0} (e0) + Up(X^{1,0} (e1))
        self.conv0_1 = ConvBlock(e_ch[0] + e_ch[1], d_ch[0])
        # X^{0,2} inputs: X^{0,0} (e0) + X^{0,1} + Up(X^{1,1})
        self.conv0_2 = ConvBlock(e_ch[0] + d_ch[0] + d_ch[1], d_ch[0])
        # X^{0,3} inputs: X^{0,0} (e0) + X^{0,1} + X^{0,2} + Up(X^{1,2})
        self.conv0_3 = ConvBlock(e_ch[0] + d_ch[0] + d_ch[0] + d_ch[1], d_ch[0])
        # X^{0,4} inputs: X^{0,0} (e0) + X^{0,1} + X^{0,2} + X^{0,3} + Up(X^{1,3})
        self.conv0_4 = ConvBlock(
            e_ch[0] + d_ch[0] + d_ch[0] + d_ch[0] + d_ch[1], d_ch[0]
        )

        # 3. Segmentation Heads (1x1 Convs)
        self.final0_1 = nn.Conv2d(d_ch[0], 1, 1)
        self.final0_2 = nn.Conv2d(d_ch[0], 1, 1)
        self.final0_3 = nn.Conv2d(d_ch[0], 1, 1)
        self.final0_4 = nn.Conv2d(d_ch[0], 1, 1)

    def forward(self, x, deep_supervision=None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            deep_supervision: Boolean to override default behavior.
                              If True, returns list of 4 outputs.
                              If False, returns single final output.
        """
        if deep_supervision is None:
            deep_supervision = self.deep_supervision

        # Encoder Pass
        # e0: 64x64, e1: 32x32, e2: 16x16, e3: 8x8, e4: 4x4 (assuming 128x128 input)
        e0, e1, e2, e3, e4 = self.encoder(x)

        # --- Decoder Nested Pass ---

        # Helper for upsampling
        def up(t):
            return F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=True)

        # Column 1 (j=1)
        x3_1 = self.conv3_1(torch.cat([e3, up(e4)], dim=1))
        x2_1 = self.conv2_1(torch.cat([e2, up(e3)], dim=1))
        x1_1 = self.conv1_1(torch.cat([e1, up(e2)], dim=1))
        x0_1 = self.conv0_1(torch.cat([e0, up(e1)], dim=1))

        # Column 2 (j=2)
        x2_2 = self.conv2_2(torch.cat([e2, x2_1, up(x3_1)], dim=1))
        x1_2 = self.conv1_2(torch.cat([e1, x1_1, up(x2_1)], dim=1))
        x0_2 = self.conv0_2(torch.cat([e0, x0_1, up(x1_1)], dim=1))

        # Column 3 (j=3)
        x1_3 = self.conv1_3(torch.cat([e1, x1_1, x1_2, up(x2_2)], dim=1))
        x0_3 = self.conv0_3(torch.cat([e0, x0_1, x0_2, up(x1_2)], dim=1))

        # Column 4 (j=4) - Final Output Node
        x0_4 = self.conv0_4(torch.cat([e0, x0_1, x0_2, x0_3, up(x1_3)], dim=1))

        # Generate Final Output (Upsample from Stride 2 to Stride 1)
        out4 = up(self.final0_4(x0_4))

        if deep_supervision:
            # Generate intermediate outputs for deep supervision
            out1 = up(self.final0_1(x0_1))
            out2 = up(self.final0_2(x0_2))
            out3 = up(self.final0_3(x0_3))
            return [out1, out2, out3, out4]
        else:
            return out4
