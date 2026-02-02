import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for shape (N, C, H, W).
    Standard nn.LayerNorm expects (N, ..., C), so this handles the permutation.
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
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted for Decoder usage.
    Structure: 7x7 Depthwise -> LayerNorm -> 1x1 Pointwise -> GELU -> 1x1 Pointwise
    Maintains the large receptive field and linearity properties.
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # We use 1x1 convs instead of Linear for NCHW implementation convenience
        # to avoid excessive permuting, or we can permute.
        # To strictly follow ConvNeXt design which uses Linear on channels:
        self.pwconv1_conv = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.pwconv2_conv = nn.Conv2d(4 * dim, dim, kernel_size=1)

        self.drop_path = (
            nn.Identity()
        )  # Placeholder, usually not needed for decoder fine-tuning

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1_conv(x)
        x = self.act(x)
        x = self.pwconv2_conv(x)
        x = input + self.drop_path(x)
        return x


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block.
    Concurrent Spatial and Channel attention to suppress noise in skip connections.
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
    Captures multi-scale context at the bottleneck.
    """

    def __init__(self, in_channels, out_channels, rates=[1, 6, 12, 18]):
        super().__init__()
        modules = []
        # 1x1 Conv
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )
        # Atrous Convolutions
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
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            # Handle global pooling branch upsampling
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
    Sequential Decoder Block.
    Upsample -> Concat (Optional) -> SCSE (Optional) -> Conv -> ConvNeXtBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        # If we have a skip connection, we concat, so input dim increases
        self.has_skip = skip_channels > 0
        concat_channels = in_channels + skip_channels

        if self.has_skip:
            self.scse = SCSEModule(concat_channels)

        # Reduction / Fusion conv
        self.conv = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Refinement with Large Kernel
        self.block = ConvNeXtBlock(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)

        if self.has_skip and skip is not None:
            # Ensure dimensions match (handle potential rounding errors in deep networks)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = self.scse(x)

        x = self.conv(x)
        x = self.block(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    Raw-Physics ConvNeXt U-Net with Focal-Batch Optimization.

    Encoder: ConvNeXt-Tiny (6-channel input, pretrained weights averaged)
    Bottleneck: ASPP
    Decoder: Sequential U-Net with ConvNeXt Blocks and SCSE Attention
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder: ConvNeXt Tiny
        # We load features_only to get the feature pyramid
        # out_indices: 0 (stride 4), 1 (stride 8), 2 (stride 16), 3 (stride 32)
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
            out_indices=(0, 1, 2, 3),
        )

        # Channel counts for ConvNeXt-Tiny: [96, 192, 384, 768]
        dims = self.encoder.feature_info.channels()

        # 3. Bottleneck: ASPP
        self.aspp = ASPP(in_channels=dims[3], out_channels=dims[3])

        # 4. Decoder
        # Stride 32 -> 16
        self.up1 = DecoderBlock(
            in_channels=dims[3], skip_channels=dims[2], out_channels=dims[2]
        )
        # Stride 16 -> 8
        self.up2 = DecoderBlock(
            in_channels=dims[2], skip_channels=dims[1], out_channels=dims[1]
        )
        # Stride 8 -> 4
        self.up3 = DecoderBlock(
            in_channels=dims[1], skip_channels=dims[0], out_channels=dims[0]
        )
        # Stride 4 -> 2 (No skip from encoder, as stem is stride 4)
        self.up4 = DecoderBlock(
            in_channels=dims[0], skip_channels=0, out_channels=dims[0] // 2
        )
        # Stride 2 -> 1
        self.up5 = DecoderBlock(
            in_channels=dims[0] // 2, skip_channels=0, out_channels=dims[0] // 4
        )

        # 5. Head
        self.final_conv = nn.Conv2d(dims[0] // 4, Config.NUM_CLASSES, kernel_size=1)

    def forward(self, x):
        # Encoder
        # enc_feats: [stride 4, stride 8, stride 16, stride 32]
        enc_feats = self.encoder(x)

        # Bottleneck
        x = self.aspp(enc_feats[3])

        # Decoder
        x = self.up1(x, enc_feats[2])  # 32 -> 16
        x = self.up2(x, enc_feats[1])  # 16 -> 8
        x = self.up3(x, enc_feats[0])  # 8 -> 4
        x = self.up4(x)  # 4 -> 2
        x = self.up5(x)  # 2 -> 1

        # Head
        logits = self.final_conv(x)

        # Ensure output size matches input size exactly (256x256)
        # (Sometimes padding in convs can cause slight mismatches)
        if logits.shape[2:] != (Config.IMG_SIZE, Config.IMG_SIZE):
            logits = F.interpolate(
                logits,
                size=(Config.IMG_SIZE, Config.IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            )

        return logits
