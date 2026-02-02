import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for channel-first tensors (N, C, H, W).
    ConvNeXt uses LayerNorm, but standard nn.LayerNorm expects (N, ..., C).
    """

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted for the decoder.
    Uses 7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv.
    Maintains the large kernel isotropic prior.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with Linear
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            nn.Identity()
        )  # Placeholder, usually handled by timm's DropPath in training

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)

        # Permute for linear layers: (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    Improves data efficiency by recalibrating features after concatenation.
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


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Captures multi-scale global context at the bottleneck.
    """

    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
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
        for rate in atrous_rates:
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

        # Global Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Project after concatenation
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # Handle global pooling upsampling
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
    Decoder block combining Upsampling, Skip Connection, SCSE, and Large-Kernel Refinement.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        # Calculate intermediate channels after concatenation
        concat_channels = in_channels + skip_channels

        self.scse = SCSEModule(concat_channels)

        # Use ConvNeXt block for refinement to maintain large receptive field
        self.block = ConvNeXtBlock(concat_channels)

        # Project to desired output channels
        self.project = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.scse(x)
        x = self.block(x)
        x = self.project(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    Isotropic Large-Kernel ConvNeXt U-Net with Global Context and Attention.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (ConvNeXt Tiny)
        # in_chans=6 for Ash + Temporal Difference
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            in_chans=Config.IN_CHANNELS,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # ConvNeXt Tiny feature channels: [96, 192, 384, 768]
        # Corresponding strides: [4, 8, 16, 32]
        enc_channels = [96, 192, 384, 768]

        # 2. Bottleneck (ASPP)
        # Input: 768 (Stride 32), Output: 256
        self.aspp = ASPP(enc_channels[3], Config.DECODER_CHANNELS[0])

        # 3. Decoder
        # Config.DECODER_CHANNELS = [256, 128, 64, 32, 16]

        # Decoder 0: Stride 32 -> 16. Skip: Stride 16 (384 channels)
        self.decoder0 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[0],
            skip_channels=enc_channels[2],
            out_channels=Config.DECODER_CHANNELS[0],
        )

        # Decoder 1: Stride 16 -> 8. Skip: Stride 8 (192 channels)
        self.decoder1 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[0],
            skip_channels=enc_channels[1],
            out_channels=Config.DECODER_CHANNELS[1],
        )

        # Decoder 2: Stride 8 -> 4. Skip: Stride 4 (96 channels)
        self.decoder2 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[1],
            skip_channels=enc_channels[0],
            out_channels=Config.DECODER_CHANNELS[2],
        )

        # Decoder 3: Stride 4 -> 2. No Skip (or input, but we just upsample)
        # We treat skip_channels=0
        self.decoder3 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[2],
            skip_channels=0,
            out_channels=Config.DECODER_CHANNELS[3],
        )

        # Decoder 4: Stride 2 -> 1. No Skip.
        self.decoder4 = DecoderBlock(
            in_channels=Config.DECODER_CHANNELS[3],
            skip_channels=0,
            out_channels=Config.DECODER_CHANNELS[4],
        )

        # 4. Segmentation Head
        self.segmentation_head = nn.Conv2d(
            Config.DECODER_CHANNELS[4], Config.NUM_CLASSES, kernel_size=1
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features[0]: stride 4 (96)
        # features[1]: stride 8 (192)
        # features[2]: stride 16 (384)
        # features[3]: stride 32 (768)

        # Bottleneck
        x = self.aspp(features[3])  # -> 256, stride 32

        # Decoder
        x = self.decoder0(x, features[2])  # -> 256, stride 16
        x = self.decoder1(x, features[1])  # -> 128, stride 8
        x = self.decoder2(x, features[0])  # -> 64, stride 4
        x = self.decoder3(x)  # -> 32, stride 2
        x = self.decoder4(x)  # -> 16, stride 1

        # Head
        logits = self.segmentation_head(x)

        # Ensure output matches input size exactly (handles odd padding if any)
        # Usually not needed with power-of-2 sizes, but good for safety
        # logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return logits
