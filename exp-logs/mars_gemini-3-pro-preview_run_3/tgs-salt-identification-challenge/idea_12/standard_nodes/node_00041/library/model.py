import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    Ref: https://arxiv.org/abs/1803.02579
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Convolutional Block for U-Net++ Decoder Nodes.
    Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> scSE
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x


class UNetPlusPlus(nn.Module):
    """
    U-Net++ with ResNeXt-50 Encoder and Deep Supervision.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------------------
        # Using timm to load ResNeXt50_32x4d
        # Features at indices 0..4 correspond to strides 2, 4, 8, 16, 32
        self.encoder = timm.create_model(
            Config.ENCODER,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Encoder channels for ResNeXt50_32x4d: [64, 256, 512, 1024, 2048]
        e_ch = [64, 256, 512, 1024, 2048]

        # Decoder channels from Config: (256, 128, 64, 32, 16)
        # Mapping to levels:
        # L4 (s16): 256
        # L3 (s8):  128
        # L2 (s4):  64
        # L1 (s2):  32
        # L0 (s1):  16
        d_ch = Config.DECODER_CHANNELS

        # ---------------------------------------------------------------------
        # Decoder Blocks (Nested Dense Pathways)
        # ---------------------------------------------------------------------
        # Notation: conv_L_D where L is level (resolution), D is depth (dense index)

        # --- Column 1 (j=1) ---
        # Inputs: Encoder feature + Upsampled feature from below
        self.conv_4_1 = ConvBlock(
            e_ch[3] + e_ch[4], d_ch[0]
        )  # s16: [1024 + 2048] -> 256 (Wait, upsampling doesn't change ch unless conv'd)
        # Actually, we concatenate Up(X_{i+1}) with X_{i}.
        # Up(X) keeps channels of X.

        self.conv_3_1 = ConvBlock(
            e_ch[3] + d_ch[0], d_ch[0]
        )  # s16: [Enc(s16) + Up(Enc(s32)? No)]
        # Correction: U-Net++ connects X_{i,0} and Up(X_{i+1, 0})
        # My levels:
        # Row 4 (s32): Enc only (e_ch[4])
        # Row 3 (s16): Enc (e_ch[3]) + Up(Row 4)
        # Row 2 (s8):  Enc (e_ch[2]) + Up(Row 3)
        # ...

        # Re-defining based on standard U-Net++ indexing relative to encoder
        # X_{i, j}
        # i=0 (s2), i=1 (s4), i=2 (s8), i=3 (s16), i=4 (s32)

        # j=1
        self.conv_3_1 = ConvBlock(
            e_ch[3] + e_ch[4], d_ch[0]
        )  # s16: [1024 + 2048] -> 256
        self.conv_2_1 = ConvBlock(e_ch[2] + d_ch[0], d_ch[1])  # s8:  [512 + 256] -> 128
        self.conv_1_1 = ConvBlock(e_ch[1] + d_ch[1], d_ch[2])  # s4:  [256 + 128] -> 64
        self.conv_0_1 = ConvBlock(e_ch[0] + d_ch[2], d_ch[3])  # s2:  [64 + 64] -> 32

        # j=2
        self.conv_2_2 = ConvBlock(
            e_ch[2] + 2 * d_ch[1], d_ch[1]
        )  # s8:  [512 + 128 + 128] -> 128 (Enc + X_21 + Up(X_31))
        # Note: Up(X_31) is 256 ch. X_21 is 128 ch. Enc is 512.
        # Input dim = 512 + 128 + 256 = 896?
        # Correct logic: Concatenate [X_{i,0}, X_{i,1}, ..., Up(X_{i+1, j-1})]
        self.conv_2_2 = ConvBlock(e_ch[2] + d_ch[1] + d_ch[0], d_ch[1])
        self.conv_1_2 = ConvBlock(e_ch[1] + d_ch[2] + d_ch[1], d_ch[2])
        self.conv_0_2 = ConvBlock(e_ch[0] + d_ch[3] + d_ch[2], d_ch[3])

        # j=3
        self.conv_1_3 = ConvBlock(e_ch[1] + 2 * d_ch[2] + d_ch[1], d_ch[2])
        self.conv_0_3 = ConvBlock(e_ch[0] + 2 * d_ch[3] + d_ch[2], d_ch[3])

        # j=4
        self.conv_0_4 = ConvBlock(e_ch[0] + 3 * d_ch[3] + d_ch[2], d_ch[3])

        # --- Level -1 (Stride 1 / Full Resolution) ---
        # This row does not exist in Encoder. Constructed from Up(Row 0).
        # We use d_ch[4] = 16 for this level.

        # j=1 (Connected to X_0_1)
        self.conv_m1_1 = ConvBlock(d_ch[3], d_ch[4])  # In: Up(X_0_1) [32] -> Out: 16

        # j=2 (Connected to X_m1_1, Up(X_0_2))
        self.conv_m1_2 = ConvBlock(d_ch[4] + d_ch[3], d_ch[4])  # In: [16 + 32] -> 16

        # j=3 (Connected to X_m1_1, X_m1_2, Up(X_0_3))
        self.conv_m1_3 = ConvBlock(
            2 * d_ch[4] + d_ch[3], d_ch[4]
        )  # In: [16 + 16 + 32] -> 16

        # j=4 (Connected to X_m1_1, X_m1_2, X_m1_3, Up(X_0_4))
        self.conv_m1_4 = ConvBlock(
            3 * d_ch[4] + d_ch[3], d_ch[4]
        )  # In: [16 + 16 + 16 + 32] -> 16

        # ---------------------------------------------------------------------
        # Segmentation Heads (Deep Supervision)
        # ---------------------------------------------------------------------
        # We attach heads to the Full Resolution nodes: X_{-1, 1..4}
        self.head1 = nn.Conv2d(d_ch[4], 1, kernel_size=1)
        self.head2 = nn.Conv2d(d_ch[4], 1, kernel_size=1)
        self.head3 = nn.Conv2d(d_ch[4], 1, kernel_size=1)
        self.head4 = nn.Conv2d(d_ch[4], 1, kernel_size=1)

    def _upsample_add(self, x, y):
        """Upsample x and concatenate with y."""
        _, _, h, w = y.size()
        x_up = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=True)
        return torch.cat([y, x_up], dim=1)

    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=True)

    def forward(self, x):
        # Input shape: (B, 3, H, W)
        input_size = x.shape[2:]

        # Encoder
        features = self.encoder(x)
        x0_0 = features[0]  # s2
        x1_0 = features[1]  # s4
        x2_0 = features[2]  # s8
        x3_0 = features[3]  # s16
        x4_0 = features[4]  # s32

        # Decoder Column 1
        x3_1 = self.conv_3_1(self._upsample_add(x4_0, x3_0))
        x2_1 = self.conv_2_1(self._upsample_add(x3_1, x2_0))
        x1_1 = self.conv_1_1(self._upsample_add(x2_1, x1_0))
        x0_1 = self.conv_0_1(self._upsample_add(x1_1, x0_0))

        # Decoder Column 2
        x2_2 = self.conv_2_2(self._upsample_add(x3_1, torch.cat([x2_0, x2_1], 1)))
        x1_2 = self.conv_1_2(self._upsample_add(x2_1, torch.cat([x1_0, x1_1], 1)))
        x0_2 = self.conv_0_2(self._upsample_add(x1_2, torch.cat([x0_0, x0_1], 1)))

        # Decoder Column 3
        x1_3 = self.conv_1_3(self._upsample_add(x2_2, torch.cat([x1_0, x1_1, x1_2], 1)))
        x0_3 = self.conv_0_3(self._upsample_add(x1_3, torch.cat([x0_0, x0_1, x0_2], 1)))

        # Decoder Column 4
        x0_4 = self.conv_0_4(
            self._upsample_add(x1_3, torch.cat([x0_0, x0_1, x0_2, x0_3], 1))
        )

        # Level -1 (Full Resolution Construction)
        # x0_j are at stride 2. We upsample them to input_size.

        # xm1_1
        xm1_1 = self.conv_m1_1(self._upsample(x0_1, input_size))

        # xm1_2
        # Input: xm1_1 (lateral), Up(x0_2)
        xm1_2 = self.conv_m1_2(torch.cat([xm1_1, self._upsample(x0_2, input_size)], 1))

        # xm1_3
        # Input: xm1_1, xm1_2, Up(x0_3)
        xm1_3 = self.conv_m1_3(
            torch.cat([xm1_1, xm1_2, self._upsample(x0_3, input_size)], 1)
        )

        # xm1_4
        # Input: xm1_1, xm1_2, xm1_3, Up(x0_4)
        xm1_4 = self.conv_m1_4(
            torch.cat([xm1_1, xm1_2, xm1_3, self._upsample(x0_4, input_size)], 1)
        )

        # Heads
        out1 = self.head1(xm1_1)
        out2 = self.head2(xm1_2)
        out3 = self.head3(xm1_3)
        out4 = self.head4(xm1_4)

        if self.training:
            return [out1, out2, out3, out4]
        else:
            # Inference: Return the most refined output
            return out4
