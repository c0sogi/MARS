import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import INPUT_CHANNELS, NUM_CLASSES


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
    ConvNeXt Block adapted for Decoder usage.
    7x7 Depthwise -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv -> Residual
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
        self.drop_path = nn.Identity()  # Simplified for inference/decoder stability

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


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation (SCSE).
    Concurrent Spatial and Channel Squeeze & Excitation.
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

    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
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
            # Handle GAP module upsampling
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
    Decoder Block: Upsample -> Concat -> Reduce -> ConvNeXtBlock -> SCSE
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(
            in_channels + skip_channels, out_channels, 1, bias=False
        )
        self.block = ConvNeXtBlock(out_channels)
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
        x = self.block(x)
        x = self.scse(x)
        return x


class IsotropicConvNeXtUNet(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()

        # 1. Encoder: ConvNeXt Tiny
        # features_only=True returns a list of feature maps
        # We pass in_chans to let timm handle the first layer adaptation (Cite debug_lesson_7)
        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=True,
            features_only=True,
            in_chans=input_channels,
        )

        # Encoder Channel Dimensions (ConvNeXt Tiny)
        # dims: [96, 192, 384, 768] for indices [0, 1, 2, 3]
        # Index 0: Stage 0 (Stride 4)
        # Index 1: Stage 1 (Stride 8)
        # Index 2: Stage 2 (Stride 16)
        # Index 3: Stage 3 (Stride 32)
        enc_dims = self.encoder.feature_info.channels()

        # 2. Bridge (ASPP)
        # Input: Stride 32 (768)
        self.aspp = ASPP(enc_dims[3], 256)

        # 3. Decoder
        # Decoder 1: 32 -> 16 (Skip: Stage 2, 384)
        self.dec1 = DecoderBlock(256, enc_dims[2], 256)

        # Decoder 2: 16 -> 8 (Skip: Stage 1, 192)
        self.dec2 = DecoderBlock(256, enc_dims[1], 128)

        # Decoder 3: 8 -> 4 (Skip: Stage 0, 96)
        self.dec3 = DecoderBlock(128, enc_dims[0], 64)

        # Decoder 4: 4 -> 2 (No Skip from encoder, just upsample)
        self.dec4 = DecoderBlock(64, 0, 32)

        # Decoder 5: 2 -> 1 (No Skip)
        self.dec5 = DecoderBlock(32, 0, 16)

        # 4. Head
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        # features list indices:
        # 0: Stride 4 (Stage 0)
        # 1: Stride 8 (Stage 1)
        # 2: Stride 16 (Stage 2)
        # 3: Stride 32 (Stage 3)

        # Bridge
        x_bottleneck = self.aspp(features[3])  # (B, 256, H/32, W/32)

        # Decoder
        x = self.dec1(x_bottleneck, features[2])  # -> Stride 16
        x = self.dec2(x, features[1])  # -> Stride 8
        x = self.dec3(x, features[0])  # -> Stride 4
        x = self.dec4(x)  # -> Stride 2
        x = self.dec5(x)  # -> Stride 1

        # Head
        logits = self.head(x)

        # Ensure output size matches input size exactly (handle odd dimensions if any)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(
                logits, size=x.shape[2:], mode="bilinear", align_corners=False
            )

        return logits
