import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LayerNorm2d(nn.Module):
    """
    LayerNorm for channels_first tensors (N, C, H, W).
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
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = (
            nn.Identity()
        )  # Placeholder, usually handled by training loop or timm

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
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Module.
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
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    """

    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18]):
        super().__init__()
        self.aspp0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = F.interpolate(
            self.global_avg_pool(x),
            size=x.size()[2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat((x0, x1, x2, x3, x4), dim=1)
        return self.project(x)


class LightweightFusionHead(nn.Module):
    """
    Projects features from Stride 4, Stride 2, and Stride 1 to low dimensions,
    upsamples them, and fuses them for the final prediction.
    """

    def __init__(self, dims, proj_dim=8, num_classes=1):
        """
        dims: list of input channels [dim_s1, dim_s2, dim_s4]
        """
        super().__init__()
        self.proj_s1 = nn.Conv2d(dims[0], proj_dim, 1)
        self.proj_s2 = nn.Conv2d(dims[1], proj_dim, 1)
        self.proj_s4 = nn.Conv2d(dims[2], proj_dim, 1)

        self.classifier = nn.Conv2d(proj_dim * 3, num_classes, 1)

    def forward(self, f_s1, f_s2, f_s4):
        # Project
        p_s1 = self.proj_s1(f_s1)
        p_s2 = self.proj_s2(f_s2)
        p_s4 = self.proj_s4(f_s4)

        # Upsample to S1 size
        target_size = p_s1.shape[2:]
        p_s2_up = F.interpolate(
            p_s2, size=target_size, mode="bilinear", align_corners=False
        )
        p_s4_up = F.interpolate(
            p_s4, size=target_size, mode="bilinear", align_corners=False
        )

        # Concatenate
        fused = torch.cat([p_s1, p_s2_up, p_s4_up], dim=1)

        # Classify
        out = self.classifier(fused)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels + skip_channels, out_channels, 1)
        self.scse = SCSEModule(out_channels)
        self.block = ConvNeXtBlock(out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle potential padding issues if sizes don't match exactly
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        x = self.reduce(x)
        x = self.scse(x)
        x = self.block(x)
        return x


class ConvNeXtUNet(nn.Module):
    def __init__(
        self,
        backbone_name="convnext_tiny",
        in_channels=6,
        num_classes=1,
        pretrained=True,
    ):
        super().__init__()

        # Encoder
        # features_only=True returns features at strides [4, 8, 16, 32] for ConvNeXt
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_channels,
            features_only=True,
        )

        # Get channel counts from encoder
        # Dummy forward pass to get feature info
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, 256, 256)
            features = self.encoder(dummy)
            enc_dims = [f.shape[1] for f in features]
            # e.g., tiny: [96, 192, 384, 768] for strides [4, 8, 16, 32]

        self.enc_dims = enc_dims

        # Bridge
        bridge_dim = 256
        self.aspp = ASPP(enc_dims[3], bridge_dim)

        # Decoder Stages
        # D3: S32 -> S16. Skip: enc_dims[2]
        self.dec3 = DecoderBlock(bridge_dim, enc_dims[2], 256)

        # D2: S16 -> S8. Skip: enc_dims[1]
        self.dec2 = DecoderBlock(256, enc_dims[1], 128)

        # D1: S8 -> S4. Skip: enc_dims[0]
        self.dec1 = DecoderBlock(128, enc_dims[0], 64)

        # D0_2: S4 -> S2. No Skip.
        self.dec0_2 = DecoderBlock(64, 0, 32)

        # D0_1: S2 -> S1. No Skip.
        self.dec0_1 = DecoderBlock(32, 0, 16)

        # Fusion Head
        # Inputs: S1 (16ch), S2 (32ch), S4 (64ch)
        self.head = LightweightFusionHead(
            dims=[16, 32, 64], proj_dim=32, num_classes=num_classes
        )

    def forward(self, x):
        # Encoder
        enc_feats = self.encoder(x)
        e0, e1, e2, e3 = enc_feats  # Strides: 4, 8, 16, 32

        # Bridge
        b = self.aspp(e3)

        # Decoder
        d3 = self.dec3(b, e2)  # -> S16
        d2 = self.dec2(d3, e1)  # -> S8
        d1 = self.dec1(d2, e0)  # -> S4

        d0_2 = self.dec0_2(d1)  # -> S2
        d0_1 = self.dec0_1(d0_2)  # -> S1

        # Fusion Head
        # Pass features: S1, S2, S4
        out = self.head(d0_1, d0_2, d1)

        return out
