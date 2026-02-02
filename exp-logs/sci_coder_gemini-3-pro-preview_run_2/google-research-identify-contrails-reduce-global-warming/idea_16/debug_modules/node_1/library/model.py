import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
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


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted for U-Net decoder usage.
    Structure: 7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv.
    Maintains the linearity prior with large kernels.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim
        )  # pointwise/1x1 convs, implemented with linear layers
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
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
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


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
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
        # Atrous convs
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
            out = conv(x)
            # Handle global pooling resizing
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(
                    out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)


class DecoderBlock(nn.Module):
    """
    Decoder Stage: Upsample -> Concat -> Project -> SCSE -> ConvNeXtBlock
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        # We project the concatenated features to the target out_channels
        self.project = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels, data_format="channels_first"),
            nn.ReLU(inplace=True),
        )
        self.scse = SCSEModule(out_channels)
        self.block = ConvNeXtBlock(out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Handle potential odd dimension mismatches
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.project(x)
        x = self.scse(x)
        x = self.block(x)
        return x


class ConvNeXtUNet(nn.Module):
    """
    Isotropic Large-Kernel ConvNeXt U-Net with Decoupled Spatiotemporal Input and Pyramid Fusion.
    """

    def __init__(self):
        super().__init__()

        # 1. Backbone (Encoder)
        # ConvNeXt Tiny: Features at strides [4, 8, 16, 32]
        self.encoder = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            in_chans=Config.IN_CHANNELS,
        )

        # Get feature dimensions: [96, 192, 384, 768] for tiny
        enc_dims = self.encoder.feature_info.channels()

        # 2. Context Bridge
        self.aspp = ASPP(enc_dims[3], enc_dims[3])  # 768 -> 768

        # 3. Decoder
        # Stage 1: Stride 32 -> 16
        self.dec1 = DecoderBlock(enc_dims[3], enc_dims[2], enc_dims[2])

        # Stage 2: Stride 16 -> 8
        self.dec2 = DecoderBlock(enc_dims[2], enc_dims[1], enc_dims[1])

        # Stage 3: Stride 8 -> 4 (Output P4)
        self.dec3 = DecoderBlock(enc_dims[1], enc_dims[0], enc_dims[0])

        # Stage 4: Stride 4 -> 2 (Output P2)
        # No skip connection from encoder at Stride 2 (ConvNeXt stem is stride 4)
        self.dec4 = DecoderBlock(enc_dims[0], 0, enc_dims[0] // 2)

        # Stage 5: Stride 2 -> 1 (Output P1)
        self.dec5 = DecoderBlock(enc_dims[0] // 2, 0, enc_dims[0] // 4)

        # 4. Pyramid Fusion Head
        fusion_dim = 8
        self.proj_p4 = nn.Conv2d(enc_dims[0], fusion_dim, 1)
        self.proj_p2 = nn.Conv2d(enc_dims[0] // 2, fusion_dim, 1)
        self.proj_p1 = nn.Conv2d(enc_dims[0] // 4, fusion_dim, 1)

        # Final convolution aggregating the fused features
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                fusion_dim * 3, fusion_dim * 3, 3, padding=1, groups=fusion_dim * 3
            ),  # Lightweight Depthwise
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_dim * 3, Config.NUM_CLASSES, 1),  # Pointwise
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder
        features = self.encoder(x)
        s4, s8, s16, s32 = features  # Strides 4, 8, 16, 32

        # Bridge
        b = self.aspp(s32)

        # Decoder
        d16 = self.dec1(b, s16)
        d8 = self.dec2(d16, s8)
        d4 = self.dec3(d8, s4)  # P4 (Stride 4)
        d2 = self.dec4(d4)  # P2 (Stride 2)
        d1 = self.dec5(d2)  # P1 (Stride 1)

        # Pyramid Fusion
        # Project and Upsample P4 to Stride 1
        p4 = self.proj_p4(d4)
        p4 = F.interpolate(p4, size=input_shape, mode="bilinear", align_corners=False)

        # Project and Upsample P2 to Stride 1
        p2 = self.proj_p2(d2)
        p2 = F.interpolate(p2, size=input_shape, mode="bilinear", align_corners=False)

        # Project P1 (already Stride 1)
        p1 = self.proj_p1(d1)
        if p1.shape[-2:] != input_shape:
            p1 = F.interpolate(
                p1, size=input_shape, mode="bilinear", align_corners=False
            )

        # Concatenate
        concat = torch.cat([p4, p2, p1], dim=1)

        # Final Prediction
        out = self.final_conv(concat)

        return out
