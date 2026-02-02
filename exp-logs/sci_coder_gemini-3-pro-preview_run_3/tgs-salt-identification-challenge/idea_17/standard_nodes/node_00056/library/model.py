import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
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


class DecoderBlock(nn.Module):
    """
    Standard U-Net++ Decoder Block with scSE attention.
    Consists of:
    - Concatenation of input (upsampled) and skips
    - Conv3x3 -> BN -> ReLU
    - Conv3x3 -> BN -> ReLU
    - SCSE
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        """
        Args:
            in_channels: Channels from the upsampled lower level.
            skip_channels: List of channel counts for skip connections at this level.
            out_channels: Output channels.
        """
        super().__init__()
        # Calculate total input channels (upsampled input + all skips)
        total_in_channels = in_channels + sum(skip_channels)

        self.conv1 = nn.Conv2d(
            total_in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.attention = SCSEModule(out_channels)

    def forward(self, x, skips):
        """
        Args:
            x: Input tensor from lower level (needs upsampling).
            skips: List of tensors from the same level (skip connections).
        """
        # Upsample x to match skip dimensions
        # Assuming skips[0] has the target spatial resolution
        target_size = skips[0].shape[2:]
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=True)

        # Concatenate
        out = torch.cat([x] + skips, dim=1)

        # Convolutions
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Attention
        out = self.attention(out)
        return out


class SaltUNetPlusPlus(nn.Module):
    """
    U-Net++ with SE-ResNeXt-50 (32x4d) encoder and scSE attention.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder
        # Using timm to load pretrained backbone
        # features_only=True returns feature maps at different strides
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4),
        )

        # Encoder channels (from timm seresnext50_32x4d)
        # Indices: 0 (s2), 1 (s4), 2 (s8), 3 (s16), 4 (s32)
        # Channels: [64, 256, 512, 1024, 2048]
        encoder_channels = self.encoder.feature_info.channels()

        # Decoder channels from config: (256, 128, 64, 32, 16)
        # Mapping to levels:
        # Level 4 (s32 -> s16 output? No, usually decoder reduces stride)
        # U-Net++ Levels:
        # L0 (s2): Output 16
        # L1 (s4): Output 32
        # L2 (s8): Output 64
        # L3 (s16): Output 128
        # L4 (s32): Output 256 (This is usually the bridge/center)

        # We define the number of filters for the decoder nodes at each level
        # filters[0] corresponds to Level 0 (Stride 2), filters[4] to Level 4 (Stride 32)
        # Config DECODER_CHANNELS = (256, 128, 64, 32, 16) seems to be ordered deep->shallow or just a list.
        # Based on typical usage: 256 is deepest, 16 is shallowest.
        # filters = [16, 32, 64, 128, 256]
        self.filters = [16, 32, 64, 128, 256]

        # ---------------------------------------------------------------------
        # Decoder Blocks Construction (Nested Skip Connections)
        # Nodes: X_{i,j} where i is level (0..3), j is dense index (1..4)
        # Note: We don't necessarily process Level 4 with a decoder block unless we go deeper.
        # Standard U-Net++ usually treats the deepest encoder feature as the base for upsampling.

        # Level 3 Nodes (Stride 16)
        # X_{3,1} <- Up(X_{4,0}) + X_{3,0}
        self.conv3_1 = DecoderBlock(
            encoder_channels[4], [encoder_channels[3]], self.filters[3]
        )

        # Level 2 Nodes (Stride 8)
        # X_{2,1} <- Up(X_{3,0}) + X_{2,0} (Wait, standard U-Net++ connects X_{3,1} to X_{2,1}?)
        # Standard U-Net++: X_{i,j} receives Up(X_{i+1, j-1}) and [X_{i,0}, ..., X_{i, j-1}]

        # Correct Connectivity:
        # j=1 (Standard U-Net path)
        self.conv3_1 = DecoderBlock(
            encoder_channels[4], [encoder_channels[3]], self.filters[3]
        )
        self.conv2_1 = DecoderBlock(
            self.filters[3], [encoder_channels[2]], self.filters[2]
        )
        self.conv1_1 = DecoderBlock(
            self.filters[2], [encoder_channels[1]], self.filters[1]
        )
        self.conv0_1 = DecoderBlock(
            self.filters[1], [encoder_channels[0]], self.filters[0]
        )

        # j=2
        self.conv2_2 = DecoderBlock(
            self.filters[3], [encoder_channels[2], self.filters[2]], self.filters[2]
        )
        self.conv1_2 = DecoderBlock(
            self.filters[2], [encoder_channels[1], self.filters[1]], self.filters[1]
        )
        self.conv0_2 = DecoderBlock(
            self.filters[1], [encoder_channels[0], self.filters[0]], self.filters[0]
        )

        # j=3
        self.conv1_3 = DecoderBlock(
            self.filters[2],
            [encoder_channels[1], self.filters[1], self.filters[1]],
            self.filters[1],
        )
        self.conv0_3 = DecoderBlock(
            self.filters[1],
            [encoder_channels[0], self.filters[0], self.filters[0]],
            self.filters[0],
        )

        # j=4
        self.conv0_4 = DecoderBlock(
            self.filters[1],
            [encoder_channels[0], self.filters[0], self.filters[0], self.filters[0]],
            self.filters[0],
        )

        # ---------------------------------------------------------------------
        # Final Segmentation Heads
        # Deep Supervision on X_{0,1}, X_{0,2}, X_{0,3}, X_{0,4}
        self.final1 = nn.Conv2d(self.filters[0], 1, kernel_size=1)
        self.final2 = nn.Conv2d(self.filters[0], 1, kernel_size=1)
        self.final3 = nn.Conv2d(self.filters[0], 1, kernel_size=1)
        self.final4 = nn.Conv2d(self.filters[0], 1, kernel_size=1)

    def forward(self, x):
        # Input shape: (B, 3, H, W) -> (B, 3, 128, 128)
        input_shape = x.shape[-2:]

        # 1. Encoder Pass
        features = self.encoder(x)
        # x_0: Stride 2 (64 ch)
        # x_1: Stride 4 (256 ch)
        # x_2: Stride 8 (512 ch)
        # x_3: Stride 16 (1024 ch)
        # x_4: Stride 32 (2048 ch)
        x_0, x_1, x_2, x_3, x_4 = features

        # 2. Decoder Pass

        # --- j=1 ---
        # X_{3,1} <- Up(x_4) + x_3
        x_3_1 = self.conv3_1(x_4, [x_3])
        # X_{2,1} <- Up(x_3_1) + x_2
        x_2_1 = self.conv2_1(x_3_1, [x_2])
        # X_{1,1} <- Up(x_2_1) + x_1
        x_1_1 = self.conv1_1(x_2_1, [x_1])
        # X_{0,1} <- Up(x_1_1) + x_0
        x_0_1 = self.conv0_1(x_1_1, [x_0])

        # --- j=2 ---
        # X_{2,2} <- Up(x_3_1) + x_2 + x_2_1
        x_2_2 = self.conv2_2(x_3_1, [x_2, x_2_1])
        # X_{1,2} <- Up(x_2_2) + x_1 + x_1_1
        x_1_2 = self.conv1_2(x_2_2, [x_1, x_1_1])
        # X_{0,2} <- Up(x_1_2) + x_0 + x_0_1
        x_0_2 = self.conv0_2(x_1_2, [x_0, x_0_1])

        # --- j=3 ---
        # X_{1,3} <- Up(x_2_2) + x_1 + x_1_1 + x_1_2
        x_1_3 = self.conv1_3(x_2_2, [x_1, x_1_1, x_1_2])
        # X_{0,3} <- Up(x_1_3) + x_0 + x_0_1 + x_0_2
        x_0_3 = self.conv0_3(x_1_3, [x_0, x_0_1, x_0_2])

        # --- j=4 ---
        # X_{0,4} <- Up(x_1_3) + x_0 + x_0_1 + x_0_2 + x_0_3
        x_0_4 = self.conv0_4(x_1_3, [x_0, x_0_1, x_0_2, x_0_3])

        # 3. Heads & Upsampling
        # All heads are at Stride 2 (64x64). We upsample to 128x128.

        logits_1 = self.final1(x_0_1)
        logits_2 = self.final2(x_0_2)
        logits_3 = self.final3(x_0_3)
        logits_4 = self.final4(x_0_4)

        # Upsample to input resolution
        # Note: Using align_corners=False is standard for segmentation unless specific requirements
        # But for consistency with bilinear interpolation in libraries, False is safer.
        out_1 = F.interpolate(
            logits_1, size=input_shape, mode="bilinear", align_corners=False
        )
        out_2 = F.interpolate(
            logits_2, size=input_shape, mode="bilinear", align_corners=False
        )
        out_3 = F.interpolate(
            logits_3, size=input_shape, mode="bilinear", align_corners=False
        )
        out_4 = F.interpolate(
            logits_4, size=input_shape, mode="bilinear", align_corners=False
        )

        # Return list for Deep Supervision
        # The last element is the primary prediction
        return [out_1, out_2, out_3, out_4]
