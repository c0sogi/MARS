import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import timm
from library.config import Config

# =========================================================================
# Common Helper Blocks
# =========================================================================


class DoubleConv(nn.Module):
    """
    (Conv -> BN -> ReLU) * 2
    Standard building block for U-Net decoders.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


# =========================================================================
# Stage 1: Coarse Model (GhostUNet)
# =========================================================================


class GhostModule(nn.Module):
    """
    GhostModule from 'GhostNet: More Features from Cheap Operations'.
    Splits convolution into primary conv and cheap depthwise conv.
    """

    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25):
        super(SqueezeExcite, self).__init__()
        reduced_chs = int(in_chs * se_ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chs, reduced_chs, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_chs, in_chs, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class GhostBottleneck(nn.Module):
    """
    GhostNet Bottleneck.
    """

    def __init__(
        self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1, se_ratio=0.0
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # Shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1. Ghost expansion
        x = self.ghost1(x)

        # 2. Depthwise conv
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # 3. SE
        if self.se is not None:
            x = self.se(x)

        # 4. Ghost projection
        x = self.ghost2(x)

        # 5. Shortcut
        x += self.shortcut(residual)
        return x


class GhostNetEncoder(nn.Module):
    """
    A simplified GhostNet-like encoder that returns features at 5 scales.
    Strides: 2, 4, 8, 16, 32
    """

    def __init__(self, in_chans=3, width=1.0):
        super(GhostNetEncoder, self).__init__()
        # Config for GhostNet layers: [kernel, exp_size, out_chs, se_ratio, stride]
        # Adapted to ensure we get features at specific strides
        self.cfgs = [
            # Stage 0 (Stride 2)
            [[3, 16, 16, 0, 1]],
            # Stage 1 (Stride 4)
            [[3, 48, 24, 0, 2], [3, 72, 24, 0, 1]],
            # Stage 2 (Stride 8)
            [[5, 72, 40, 0.25, 2], [5, 120, 40, 0.25, 1]],
            # Stage 3 (Stride 16)
            [
                [3, 240, 80, 0, 2],
                [3, 200, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
            ],
            # Stage 4 (Stride 32)
            [
                [5, 672, 160, 0.25, 2],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
            ],
        ]

        # Stem
        output_channel = int(16 * width)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_chans, output_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
        self.input_channel = output_channel

        self.stages = nn.ModuleList([])

        # Build stages
        for stage_cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in stage_cfg:
                output_channel = int(c * width)
                hidden_channel = int(exp_size * width)
                layers.append(
                    GhostBottleneck(
                        self.input_channel,
                        hidden_channel,
                        output_channel,
                        k,
                        s,
                        se_ratio,
                    )
                )
                self.input_channel = output_channel
            self.stages.append(nn.Sequential(*layers))

    def forward(self, x):
        features = []
        x = self.conv_stem(x)
        features.append(x)  # Stride 2

        for stage in self.stages:
            x = stage(x)
            features.append(x)

        # We need 5 features.
        # Stem -> Stride 2 (idx 0)
        # Stage 0 -> Stride 2 (Refinement, idx 1) -> Actually Stage 0 in config is stride 1 relative to stem?
        # Let's adjust logic:
        # Stem is stride 2.
        # Config Stage 0 has stride 1 -> Output Stride 2.
        # Config Stage 1 has stride 2 -> Output Stride 4.
        # Config Stage 2 has stride 2 -> Output Stride 8.
        # Config Stage 3 has stride 2 -> Output Stride 16.
        # Config Stage 4 has stride 2 -> Output Stride 32.

        # We return features from Stage 0, 1, 2, 3, 4.
        # features[0] is stem (stride 2)
        # features[1] is stage 0 (stride 2) - we can skip stem or use stage 0
        # features[2] is stage 1 (stride 4)
        # features[3] is stage 2 (stride 8)
        # features[4] is stage 3 (stride 16)
        # features[5] is stage 4 (stride 32)

        return features[1:]  # Return stages 0 to 4


class GhostUNet(nn.Module):
    """
    Stage 1: Coarse Segmentation Model.
    Lightweight U-Net with GhostNet backbone.
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES):
        super(GhostUNet, self).__init__()
        self.encoder = GhostNetEncoder(in_chans=in_channels)

        # Channel counts based on GhostNetEncoder config with width=1.0
        # Stage 0: 16
        # Stage 1: 24
        # Stage 2: 40
        # Stage 3: 112
        # Stage 4: 160
        filters = [16, 24, 40, 112, 160]

        # Decoder
        # Up 1: 160 -> 112
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(filters[4] + filters[3], filters[3])

        # Up 2: 112 -> 40
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(filters[3] + filters[2], filters[2])

        # Up 3: 40 -> 24
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = DoubleConv(filters[2] + filters[1], filters[1])

        # Up 4: 24 -> 16
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = DoubleConv(filters[1] + filters[0], filters[0])

        # Final Up to original resolution (Stride 2 -> Stride 1)
        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_final = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[0], num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        # f1: s2, f2: s4, f3: s8, f4: s16, f5: s32
        features = self.encoder(x)
        f1, f2, f3, f4, f5 = features

        # Decoder
        x = self.up1(f5)
        x = torch.cat([x, f4], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, f3], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv4(x)

        x = self.up_final(x)
        logits = self.conv_final(x)

        return logits


# =========================================================================
# Stage 2: Fine Model (EfficientNetUNet)
# =========================================================================


class EfficientNetUNet(nn.Module):
    """
    Stage 2: Fine Segmentation Model.
    U-Net with EfficientNet-B1 backbone.
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES):
        super(EfficientNetUNet, self).__init__()

        # Load backbone
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            "efficientnet_b1", pretrained=True, features_only=True, in_chans=in_channels
        )

        # Get channel counts from the encoder
        # EfficientNet-B1 features: [s2, s4, s8, s16, s32]
        # Channels: [32, 24, 40, 112, 320]
        enc_channels = self.encoder.feature_info.channels()

        # Decoder
        # Bottleneck is the last feature map (s32)

        # Up 1: s32 -> s16
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConv(enc_channels[4] + enc_channels[3], enc_channels[3])

        # Up 2: s16 -> s8
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = DoubleConv(enc_channels[3] + enc_channels[2], enc_channels[2])

        # Up 3: s8 -> s4
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = DoubleConv(enc_channels[2] + enc_channels[1], enc_channels[1])

        # Up 4: s4 -> s2
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = DoubleConv(enc_channels[1] + enc_channels[0], enc_channels[0])

        # Final Up to original resolution (Stride 2 -> Stride 1)
        self.up_final = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.final_conv = nn.Sequential(
            nn.Conv2d(enc_channels[0], 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # f0: s2, f1: s4, f2: s8, f3: s16, f4: s32
        f0, f1, f2, f3, f4 = features

        # Decoder
        x = self.up1(f4)
        x = torch.cat([x, f3], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x, f0], dim=1)
        x = self.conv4(x)

        x = self.up_final(x)
        logits = self.final_conv(x)

        return logits
