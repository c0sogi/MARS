import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to
    (batch_size, height, width, channels) while channels_first corresponds to
    (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block.
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


class ExtendedConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with extended kernel size (11x11) for the decoder.
    This large kernel size helps in reconstructing linear features (contrails).
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise Conv with 11x11 kernel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        # Pointwise Convs (implemented as Linear layers for channel-last format)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            timm.layers.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling to capture multi-scale context.
    """

    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super().__init__()
        modules = []
        # 1x1 conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        # Atrous convolutions
        for rate in rates[1:]:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=rate,
                        dilation=rate,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        # Global Average Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # Handle Global Pooling branch which needs upsampling
            out = conv(x)
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(
                    out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder block combining Upsampling, Skip Connection, Channel Reduction,
    Extended ConvNeXt Block, and SCSE Attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # 1x1 Conv to fuse and reduce channels
        self.reduce = nn.Conv2d(
            in_channels + skip_channels, out_channels, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(
            out_channels
        )  # Normalize before entering the residual block
        self.act = nn.ReLU(inplace=True)

        # Extended ConvNeXt Block with 11x11 kernel
        self.block = ExtendedConvNeXtBlock(out_channels)

        # Attention
        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip=None):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.reduce(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.block(x)
        x = self.scse(x)
        return x


class ExtendedConvNeXtUNet(nn.Module):
    """
    U-Net with ConvNeXt-Tiny Encoder, ASPP Bridge, and Extended Kernel Decoder.
    """

    def __init__(self, in_channels=Config.IN_CHANNELS, num_classes=1):
        super().__init__()

        # 1. Encoder
        # Using ConvNeXt Tiny. features_only=True returns features from stages 0-3.
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            # out_indices=(0, 1, 2, 3) -> Strides 4, 8, 16, 32
        )

        # Modify first layer for 6 channels (3 Ash + 3 Temporal)
        # ConvNeXt Tiny stem: 4x4 conv, stride 4
        stem_conv = self.encoder.stem[0]
        new_stem_conv = nn.Conv2d(
            in_channels,
            stem_conv.out_channels,
            kernel_size=stem_conv.kernel_size,
            stride=stem_conv.stride,
            padding=stem_conv.padding,
        )

        # Initialize new weights
        # Copy weights from first 3 channels to next 3 channels, divide by 2 to preserve scale
        with torch.no_grad():
            new_stem_conv.weight[:, :3] = stem_conv.weight
            new_stem_conv.weight[:, 3:] = stem_conv.weight
            new_stem_conv.weight *= 0.5
            if stem_conv.bias is not None:
                new_stem_conv.bias = stem_conv.bias

        self.encoder.stem[0] = new_stem_conv

        # Encoder Channels: [96, 192, 384, 768] for Tiny
        enc_dims = self.encoder.feature_info.channels()

        # 2. Bridge (ASPP)
        # Input: Stage 4 (Stride 32), Output: 256 channels
        self.aspp = ASPP(enc_dims[-1], 256)

        # 3. Decoder
        # Config DECODER_CHANNELS = [256, 128, 64, 32, 16]

        # Stage 1: In=256(ASPP), Skip=384(s16), Out=256
        self.dec1 = DecoderBlock(256, enc_dims[2], 256)

        # Stage 2: In=256, Skip=192(s8), Out=128
        self.dec2 = DecoderBlock(256, enc_dims[1], 128)

        # Stage 3: In=128, Skip=96(s4), Out=64
        self.dec3 = DecoderBlock(128, enc_dims[0], 64)

        # Stage 4: In=64, Skip=0, Out=32 (Upsample to s2)
        self.dec4 = DecoderBlock(64, 0, 32)

        # Stage 5: In=32, Skip=0, Out=16 (Upsample to s1)
        self.dec5 = DecoderBlock(32, 0, 16)

        # 4. Head
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features[0]: s4 (96)
        # features[1]: s8 (192)
        # features[2]: s16 (384)
        # features[3]: s32 (768)

        # Bridge
        x = self.aspp(features[3])  # (N, 256, H/32, W/32)

        # Decoder
        x = self.dec1(x, features[2])  # -> s16
        x = self.dec2(x, features[1])  # -> s8
        x = self.dec3(x, features[0])  # -> s4
        x = self.dec4(x)  # -> s2
        x = self.dec5(x)  # -> s1

        # Head
        logits = self.head(x)

        return logits
