import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LayerNorm2d(nn.Module):
    """
    Layer Normalization for (N, C, H, W) tensors.
    Normalizes over the channel dimension for each spatial location.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block adapted for the Decoder.
    Uses 7x7 Depthwise Conv -> LayerNorm -> 1x1 Conv (Expand) -> GELU -> 1x1 Conv (Contract).
    """

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim
        )  # depthwise conv
        self.norm = LayerNorm2d(dim, eps=1e-6)
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

        # Permute for Linear layers: (N, C, H, W) -> (N, H, W, C)
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
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling Module.
    """

    def __init__(self, in_channels, out_channels, dilations=[1, 6, 12, 18]):
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

        # 3x3 convs with dilation
        for dilation in dilations[1:]:
            modules.append(
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

        # Image Pooling
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)

        # Project
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
            # Handle image pooling upsampling
            if out.shape[2:] != x.shape[2:]:
                out = F.interpolate(
                    out, size=x.shape[2:], mode="bilinear", align_corners=False
                )
            res.append(out)
        res = torch.cat(res, dim=1)
        return self.project(res)


class MacroContextUNet(nn.Module):
    """
    Macro-Context ConvNeXt U-Net with Deep Supervision.

    Features:
    - Backbone: ConvNeXt-Tiny (Encoder)
    - Bridge: ASPP
    - Decoder: Large Kernel (7x7) ConvNeXt Blocks
    - Attention: SCSE on skip connections
    - Deep Supervision: Auxiliary heads at Stride 8 and Stride 4
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Encoder: ConvNeXt Tiny
        # in_chans=6 for 6-channel input (Ash + Temporal Difference)
        # features_only=True returns features at strides 4, 8, 16, 32
        self.encoder = timm.create_model(
            "convnext_tiny",
            pretrained=self.config.pretrained,
            in_chans=self.config.input_channels,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Feature channels for ConvNeXt-Tiny: [96, 192, 384, 768]
        enc_dims = self.encoder.feature_info.channels()

        # Bridge: Stride 32
        self.aspp = ASPP(enc_dims[3], enc_dims[3] // 2)  # 768 -> 384

        # Decoder Stage 1: Stride 32 -> 16
        self.dec1_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1_proj = nn.Conv2d(
            enc_dims[3] // 2 + enc_dims[2], enc_dims[2], 1
        )  # (384+384)->384
        self.dec1_scse = SCSEModule(enc_dims[2])
        self.dec1_block = ConvNeXtBlock(enc_dims[2])

        # Decoder Stage 2: Stride 16 -> 8
        self.dec2_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2_proj = nn.Conv2d(
            enc_dims[2] + enc_dims[1], enc_dims[1], 1
        )  # (384+192)->192
        self.dec2_scse = SCSEModule(enc_dims[1])
        self.dec2_block = ConvNeXtBlock(enc_dims[1])

        # Decoder Stage 3: Stride 8 -> 4
        self.dec3_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3_proj = nn.Conv2d(
            enc_dims[1] + enc_dims[0], enc_dims[0], 1
        )  # (192+96)->96
        self.dec3_scse = SCSEModule(enc_dims[0])
        self.dec3_block = ConvNeXtBlock(enc_dims[0])

        # Decoder Stage 4: Stride 4 -> 2
        # Learned upsampling to avoid interpolation bottleneck (Cite solution_lesson_node_00039)
        self.dec4_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4_proj = nn.Conv2d(enc_dims[0], enc_dims[0] // 2, 1)
        self.dec4_block = ConvNeXtBlock(enc_dims[0] // 2)

        # Decoder Stage 5: Stride 2 -> 1
        self.dec5_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec5_proj = nn.Conv2d(enc_dims[0] // 2, enc_dims[0] // 4, 1)
        self.dec5_block = ConvNeXtBlock(enc_dims[0] // 4)

        # Heads
        # Main Head (from Stride 1)
        self.head_main = nn.Sequential(
            nn.Conv2d(enc_dims[0] // 4, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

        # Aux Head 1 (from Stride 4 - Parallel supervision)
        self.head_aux1 = nn.Conv2d(enc_dims[0], 1, 1)

        # Aux Head 2 (from Stride 8)
        self.head_aux2 = nn.Conv2d(enc_dims[1], 1, 1)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Encoder
        # features: [s4, s8, s16, s32]
        enc_feats = self.encoder(x)
        e0, e1, e2, e3 = enc_feats  # Channels: 96, 192, 384, 768

        # Bridge
        b = self.aspp(e3)  # Stride 32

        # Decoder Stage 1 (32->16)
        d1 = self.dec1_up(b)
        # Handle potential shape mismatch due to padding in encoder
        if d1.shape[-2:] != e2.shape[-2:]:
            d1 = F.interpolate(
                d1, size=e2.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([d1, e2], dim=1)
        d1 = self.dec1_proj(d1)
        d1 = self.dec1_scse(d1)
        d1 = self.dec1_block(d1)

        # Decoder Stage 2 (16->8)
        d2 = self.dec2_up(d1)
        if d2.shape[-2:] != e1.shape[-2:]:
            d2 = F.interpolate(
                d2, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_proj(d2)
        d2 = self.dec2_scse(d2)
        d2 = self.dec2_block(d2)

        # Aux Head 2 (Stride 8)
        logits_aux2 = self.head_aux2(d2)

        # Decoder Stage 3 (8->4)
        d3 = self.dec3_up(d2)
        if d3.shape[-2:] != e0.shape[-2:]:
            d3 = F.interpolate(
                d3, size=e0.shape[-2:], mode="bilinear", align_corners=False
            )
        d3 = torch.cat([d3, e0], dim=1)
        d3 = self.dec3_proj(d3)
        d3 = self.dec3_scse(d3)
        d3 = self.dec3_block(d3)

        # Aux Head 1 (Stride 4)
        logits_aux1 = self.head_aux1(d3)

        # Decoder Stage 4 (4->2)
        d4 = self.dec4_up(d3)
        d4 = self.dec4_proj(d4)
        d4 = self.dec4_block(d4)

        # Decoder Stage 5 (2->1)
        d5 = self.dec5_up(d4)
        d5 = self.dec5_proj(d5)
        d5 = self.dec5_block(d5)

        # Main Head (Stride 1)
        logits_main = self.head_main(d5)

        # Upsample aux heads to input resolution
        # Main head is already at input resolution (Stride 1)
        logits_aux1 = F.interpolate(
            logits_aux1, size=input_shape, mode="bilinear", align_corners=False
        )
        logits_aux2 = F.interpolate(
            logits_aux2, size=input_shape, mode="bilinear", align_corners=False
        )

        if self.training and self.config.deep_supervision:
            # Return list for hybrid loss calculation
            return [logits_main, logits_aux1, logits_aux2]
        else:
            return logits_main
