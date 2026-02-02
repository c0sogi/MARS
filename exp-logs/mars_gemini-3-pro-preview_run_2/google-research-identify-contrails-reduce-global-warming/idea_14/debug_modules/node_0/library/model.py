import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for (N, C, H, W) tensors.
    Normalizes across the channel dimension C for each spatial location (h, w).
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


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
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


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted for U-Net decoder.
    Uses 7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv -> GELU -> 1x1 Conv.
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs
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

        # Permute to (N, H, W, C) for Linear layers
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # Permute back to (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    """

    def __init__(self, in_channels, out_channels, atrous_rates=[1, 6, 12, 18]):
        super(ASPP, self).__init__()
        modules = []

        # 1x1 Convolution
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Atrous Convolutions
        for rate in atrous_rates:
            if rate == 1:
                continue
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

        self.convs = nn.ModuleList(modules)

        # Global Average Pooling branch
        self.global_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(
                len(modules) * out_channels + out_channels, out_channels, 1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # Global pooling branch
        gp = self.global_pooling(x)
        gp = F.interpolate(gp, size=x.shape[2:], mode="bilinear", align_corners=False)
        res.append(gp)

        res = torch.cat(res, dim=1)
        return self.project(res)


class ConvNeXtHyperUNet(nn.Module):
    """
    Hyper-Dense ConvNeXt U-Net with Multi-Scale Feature Aggregation.
    """

    def __init__(self):
        super().__init__()

        # 1. Encoder (ConvNeXt Tiny)
        # Using features_only=True to get intermediate feature maps
        self.encoder = timm.create_model(
            Config.ENCODER_NAME,
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Adapt first layer for 6-channel input
        self._modify_first_layer()

        # Get channel counts: [96, 192, 384, 768] for convnext_tiny
        enc_dims = self.encoder.feature_info.channels()

        # 2. Bottleneck (ASPP)
        # Input: Stride 32 (768 ch) -> Output: 384 ch
        self.aspp = ASPP(enc_dims[3], enc_dims[3] // 2)

        # 3. Decoder Stages

        # Stage 4: Stride 32 -> 16
        # Input: ASPP(384) + Skip e2(384) = 768
        self.up4 = nn.Sequential(
            nn.Conv2d(384 + enc_dims[2], 384, 1), ConvNeXtBlock(384), SCSEModule(384)
        )

        # Stage 3: Stride 16 -> 8 (Hyper Feature 1)
        # Input: Up4(384) + Skip e1(192) = 576
        self.up3 = nn.Sequential(
            nn.Conv2d(384 + enc_dims[1], 192, 1), ConvNeXtBlock(192), SCSEModule(192)
        )

        # Stage 2: Stride 8 -> 4 (Hyper Feature 2)
        # Input: Up3(192) + Skip e0(96) = 288
        self.up2 = nn.Sequential(
            nn.Conv2d(192 + enc_dims[0], 96, 1), ConvNeXtBlock(96), SCSEModule(96)
        )

        # Stage 1: Stride 4 -> 2 (Hyper Feature 3)
        # Input: Up2(96) -> Upsample
        self.up1 = nn.Sequential(
            nn.Conv2d(96, 48, 1), ConvNeXtBlock(48), SCSEModule(48)
        )

        # Stage 0: Stride 2 -> 1 (Hyper Feature 4)
        # Input: Up1(48) -> Upsample
        self.up0 = nn.Sequential(
            nn.Conv2d(48, 24, 1), ConvNeXtBlock(24), SCSEModule(24)
        )

        # 4. Hypercolumn Aggregation Head
        # Concatenates features from Strides 8, 4, 2, 1
        total_hyper_channels = 192 + 96 + 48 + 24

        self.hyper_head = nn.Sequential(
            ConvNeXtBlock(total_hyper_channels),
            nn.Conv2d(total_hyper_channels, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, Config.OUT_CHANNELS, 1),
        )

    def _modify_first_layer(self):
        """
        Modifies the first convolution layer of the encoder to accept IN_CHANNELS (6).
        Copies weights from the first 3 channels to the new channels.
        """
        old_conv = self.encoder.stem[0]
        new_conv = nn.Conv2d(
            Config.IN_CHANNELS,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            # Copy original weights to first 3 channels
            new_conv.weight[:, :3] = old_conv.weight
            # Copy original weights to next 3 channels (simple initialization)
            new_conv.weight[:, 3:] = old_conv.weight
            new_conv.bias = old_conv.bias

        self.encoder.stem[0] = new_conv

    def forward(self, x):
        # --- Encoder ---
        # e0: Stride 4, e1: Stride 8, e2: Stride 16, e3: Stride 32
        enc_feats = self.encoder(x)
        e0, e1, e2, e3 = enc_feats

        # --- Bottleneck ---
        b = self.aspp(e3)  # Stride 32

        # --- Decoder ---

        # Up to Stride 16
        d4 = F.interpolate(b, scale_factor=2, mode="bilinear", align_corners=False)
        d4 = torch.cat([d4, e2], dim=1)
        d4 = self.up4(d4)

        # Up to Stride 8 (Hyper Feature 1)
        d3 = F.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.up3(d3)

        # Up to Stride 4 (Hyper Feature 2)
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e0], dim=1)
        d2 = self.up2(d2)

        # Up to Stride 2 (Hyper Feature 3)
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up1(d1)

        # Up to Stride 1 (Hyper Feature 4)
        d0 = F.interpolate(d1, scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.up0(d0)

        # --- Hypercolumn Aggregation ---
        # Upsample all relevant feature maps to Stride 1 (Output Size)
        target_size = d0.shape[2:]

        h1 = F.interpolate(
            d3, size=target_size, mode="bilinear", align_corners=False
        )  # From Stride 8
        h2 = F.interpolate(
            d2, size=target_size, mode="bilinear", align_corners=False
        )  # From Stride 4
        h3 = F.interpolate(
            d1, size=target_size, mode="bilinear", align_corners=False
        )  # From Stride 2
        h4 = d0  # From Stride 1

        # Concatenate
        hyper_col = torch.cat([h1, h2, h3, h4], dim=1)

        # Final Prediction
        out = self.hyper_head(hyper_col)

        return out
