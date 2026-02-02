import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block.
    Enhances important features by recalibrating channel and spatial responses.
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
        # Concurrent Spatial and Channel SE
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures multi-scale context using dilated convolutions.
    """

    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
        super().__init__()
        self.modules_list = nn.ModuleList()

        # 1x1 Conv
        self.modules_list.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated Convs
        for dilation in dilations[1:]:
            self.modules_list.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        # Image Pooling (Global context)
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for mod in self.modules_list:
            res.append(mod(x))

        # Image pooling needs upsampling to match spatial dims
        pool = self.image_pooling(x)
        pool = F.interpolate(
            pool, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        res.append(pool)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ProgressiveDecoderBlock(nn.Module):
    """
    Decoder block with Progressive Kernel Expansion.
    Uses Depthwise Separable Convolutions with large kernels to maintain physical receptive field.
    """

    def __init__(self, in_channels, skip_channels, out_channels, kernel_size):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Calculate padding to maintain dimensions (same padding)
        padding = kernel_size // 2

        concat_channels = in_channels + skip_channels

        # Depthwise Separable Convolution with Large Kernel
        # Depthwise: groups=in_channels, kernel=k
        # Pointwise: kernel=1
        self.conv = nn.Sequential(
            nn.Conv2d(
                concat_channels,
                concat_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=concat_channels,
                bias=False,
            ),
            nn.BatchNorm2d(concat_channels),
            nn.GELU(),
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)

        if skip is not None:
            # Handle potential slight shape mismatch due to odd padding in encoder
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        x = self.scse(x)
        return x


class ProgressiveConvNeXtUNet(nn.Module):
    """
    U-Net with ConvNeXt Backbone and Progressive Large-Kernel Decoder.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone: ConvNeXt Tiny
        # features_only=True returns feature maps from stages
        # out_indices=(0, 1, 2, 3) corresponds to strides 4, 8, 16, 32
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.ENCODER_PRETRAINED,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=Config.IN_CHANNELS,
        )

        # Feature dimensions for ConvNeXt Tiny: [96, 192, 384, 768]
        dims = self.encoder.feature_info.channels()

        # 2. Bottleneck: ASPP
        # Input: s32 features (768), Output: 256
        self.aspp = ASPP(dims[3], 256)

        # 3. Decoder Stages
        # Kernel sizes: [7, 9, 11, 13, 15] (from Config)
        k_sizes = Config.DECODER_KERNEL_SIZES

        # Stage 1: s32 -> s16 (Skip s16: dims[2]=384)
        self.dec1 = ProgressiveDecoderBlock(256, dims[2], 256, k_sizes[0])

        # Stage 2: s16 -> s8 (Skip s8: dims[1]=192)
        self.dec2 = ProgressiveDecoderBlock(256, dims[1], 128, k_sizes[1])

        # Stage 3: s8 -> s4 (Skip s4: dims[0]=96)
        self.dec3 = ProgressiveDecoderBlock(128, dims[0], 64, k_sizes[2])

        # Stage 4: s4 -> s2 (No skip)
        self.dec4 = ProgressiveDecoderBlock(64, 0, 32, k_sizes[3])

        # Stage 5: s2 -> s1 (No skip)
        self.dec5 = ProgressiveDecoderBlock(32, 0, 16, k_sizes[4])

        # 4. Segmentation Head
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        # features: [s4, s8, s16, s32]
        enc_feats = self.encoder(x)
        s4, s8, s16, s32 = enc_feats

        # Bottleneck
        x = self.aspp(s32)

        # Decoder
        x = self.dec1(x, s16)  # -> s16
        x = self.dec2(x, s8)  # -> s8
        x = self.dec3(x, s4)  # -> s4
        x = self.dec4(x)  # -> s2
        x = self.dec5(x)  # -> s1

        # Head
        x = self.head(x)

        return x
