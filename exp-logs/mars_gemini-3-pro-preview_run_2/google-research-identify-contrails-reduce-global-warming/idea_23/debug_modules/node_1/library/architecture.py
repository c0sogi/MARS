import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class LayerNorm2d(nn.Module):
    """
    LayerNorm that supports NCHW tensors directly (equivalent to GroupNorm(1, C)).
    ConvNeXt typically uses LayerNorm, but operating in NCHW avoids frequent permutations.
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


class ExtendedConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block with an Extended Kernel Size (11x11).
    Used in the decoder to maximize the receptive field for linear feature reconstruction.

    Structure:
    - DwConv 11x11
    - LayerNorm
    - 1x1 Conv (Expand 4x)
    - GELU
    - 1x1 Conv (Project 1x)
    - Residual Connection
    """

    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        # Extended Kernel: 11x11
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=11, padding=5, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        # Note: DropPath is omitted for simplicity in decoder usage,
        # as regularization is less critical in the upsampling path compared to encoder.

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x + shortcut
        return x


class SCSEModule(nn.Module):
    """
    Spatial and Channel Squeeze & Excitation Block.
    Enhances important features and suppresses noise.
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

        # Project combined features
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


class MetadataProjector(nn.Module):
    """
    Projects scalar metadata (lat, lon, time) into a feature vector.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),  # Normalize activation for gating/conditioning
        )

    def forward(self, x):
        # x: (B, input_dim)
        return self.mlp(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder Block with Extended ConvNeXt Refinement.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # Reduction conv to merge upsampled and skip features
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.scse = SCSEModule(out_channels)
        self.block = ExtendedConvNeXtBlock(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Handle potential padding issues if shapes don't match exactly
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        x = self.scse(x)
        x = self.block(x)
        return x


class GC_ConvNeXtUNet(nn.Module):
    """
    Global-Conditioned Extended-Kernel ConvNeXt U-Net.

    Args:
        in_chans (int): Number of input image channels.
        num_classes (int): Number of output classes (1 for binary segmentation).
        metadata_dim (int): Number of metadata features (row_min, col_min, timestamp).
    """

    def __init__(
        self,
        in_chans=Config.IN_CHANNELS,
        num_classes=1,
        metadata_dim=Config.METADATA_FEATURE_DIM,
    ):
        super().__init__()

        # 1. Encoder: ConvNeXt-Tiny
        # features_only=True returns a list of feature maps
        self.encoder = timm.create_model(
            "convnext_tiny", pretrained=True, in_chans=in_chans, features_only=True
        )

        # Encoder channels for ConvNeXt-Tiny: [96, 192, 384, 768]
        # Strides: [4, 8, 16, 32]
        enc_channels = [96, 192, 384, 768]

        # 2. Context Bridge
        self.aspp = ASPP(enc_channels[3], 256)

        # Metadata Injection
        self.meta_proj_dim = 32
        self.meta_projector = MetadataProjector(metadata_dim, self.meta_proj_dim)

        # Bridge output channels = ASPP out + Metadata out
        bridge_out_channels = 256 + self.meta_proj_dim

        # 3. Decoder
        # Dec3: Stride 32 -> 16. Skip: enc_channels[2] (384)
        self.dec3 = DecoderBlock(bridge_out_channels, enc_channels[2], 256)

        # Dec2: Stride 16 -> 8. Skip: enc_channels[1] (192)
        self.dec2 = DecoderBlock(256, enc_channels[1], 128)

        # Dec1: Stride 8 -> 4. Skip: enc_channels[0] (96)
        self.dec1 = DecoderBlock(128, enc_channels[0], 64)

        # 4. Final Upsampling (Stride 4 -> 1)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x, metadata):
        """
        Args:
            x (torch.Tensor): Image input (B, C, H, W)
            metadata (torch.Tensor): Normalized metadata (B, 3)
        """
        # --- Encoder ---
        # features is a list of tensors
        features = self.encoder(x)
        # s0: stride 4 (96), s1: stride 8 (192), s2: stride 16 (384), s3: stride 32 (768)
        s0, s1, s2, s3 = features[0], features[1], features[2], features[3]

        # --- Bridge ---
        # ASPP on high-level features
        x_aspp = self.aspp(s3)

        # Metadata Injection
        # Project metadata: (B, meta_dim) -> (B, proj_dim)
        meta_feat = self.meta_projector(metadata)
        # Broadcast to spatial dimensions of bottleneck: (B, proj_dim, H/32, W/32)
        meta_feat = (
            meta_feat.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, x_aspp.shape[2], x_aspp.shape[3])
        )

        # Concatenate
        x_bridge = torch.cat([x_aspp, meta_feat], dim=1)

        # --- Decoder ---
        x = self.dec3(x_bridge, s2)  # -> Stride 16
        x = self.dec2(x, s1)  # -> Stride 8
        x = self.dec1(x, s0)  # -> Stride 4

        # --- Head ---
        logits = self.final_up(x)  # -> Stride 1

        return logits
