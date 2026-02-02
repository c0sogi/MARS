import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Work with any number of dimensions, assuming batch is dim 0
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()

        # Channel Squeeze (Spatial Excitation)
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial Squeeze (Channel Excitation)
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ResBlock(nn.Module):
    """
    Residual Block with DropPath.
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> DropPath -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1, drop_path=0.0):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.drop_path(out)
        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    """
    Decoder block with Bilinear Upsampling, Concatenation, Convolutions, and scSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # We concatenate upsampled input with skip connection
        total_in_channels = in_channels + skip_channels

        self.conv1 = nn.Conv2d(
            total_in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attention = SCSEModule(out_channels) if Config.USE_SCSE else nn.Identity()

    def forward(self, x, skip):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle potential size mismatch due to odd dimensions (though padding to 128x128 avoids this)
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(
                x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.attention(x)
        return x


class SaltUNet(nn.Module):
    """
    High-Capacity Deep Residual U-Net with Stochastic Depth.
    """

    def __init__(self):
        super(SaltUNet, self).__init__()

        filters = Config.ENCODER_FILTERS  # e.g., [64, 128, 256, 512, 1024]
        drop_rate = Config.DROP_PATH_RATE

        # --- Encoder ---
        # Input: Image (1 ch) + Depth (1 ch) = 2 channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, filters[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder Blocks
        # We use a simple strategy: One ResBlock per level.
        # Stride 2 for downsampling in the block itself.
        self.enc1 = ResBlock(filters[0], filters[0], stride=1, drop_path=drop_rate)
        self.enc2 = ResBlock(filters[0], filters[1], stride=2, drop_path=drop_rate)
        self.enc3 = ResBlock(filters[1], filters[2], stride=2, drop_path=drop_rate)
        self.enc4 = ResBlock(filters[2], filters[3], stride=2, drop_path=drop_rate)

        # Bottleneck
        self.center = ResBlock(filters[3], filters[4], stride=2, drop_path=drop_rate)

        # --- Decoder ---
        # Dec 4: Input (1024) + Skip (512) -> 512
        self.dec4 = DecoderBlock(filters[4], filters[3], filters[3])

        # Dec 3: Input (512) + Skip (256) -> 256
        self.dec3 = DecoderBlock(filters[3], filters[2], filters[2])

        # Dec 2: Input (256) + Skip (128) -> 128
        self.dec2 = DecoderBlock(filters[2], filters[1], filters[1])

        # Dec 1: Input (128) + Skip (64) -> 64
        self.dec1 = DecoderBlock(filters[1], filters[0], filters[0])

        # --- Heads ---
        self.final_conv = nn.Conv2d(filters[0], 1, kernel_size=1)

        self.deep_supervision = Config.DEEP_SUPERVISION
        if self.deep_supervision:
            # Aux head at resolution 64x64 (output of dec2, which is 128 channels)
            self.aux_head_64 = nn.Conv2d(filters[1], 1, kernel_size=1)
            # Aux head at resolution 32x32 (output of dec3, which is 256 channels)
            self.aux_head_32 = nn.Conv2d(filters[2], 1, kernel_size=1)

    def forward(self, images, depths):
        """
        Args:
            images: (B, 1, H, W)
            depths: (B,) or (B, 1) - Depth values
        """
        # --- Input Fusion ---
        # Normalize depth simply by dividing by 1000 (approx max depth) or using stats if available.
        # Here we assume depths are raw values.
        # Create a depth channel
        b, c, h, w = images.shape
        depth_channel = (depths.view(b, 1, 1, 1).float() / 1000.0).expand(b, 1, h, w)
        x = torch.cat([images, depth_channel], dim=1)

        # --- Encoder ---
        x = self.input_conv(x)  # 128x128, 64ch

        e1 = self.enc1(x)  # 128x128, 64ch
        e2 = self.enc2(e1)  # 64x64, 128ch
        e3 = self.enc3(e2)  # 32x32, 256ch
        e4 = self.enc4(e3)  # 16x16, 512ch

        f = self.center(e4)  # 8x8, 1024ch

        # --- Decoder ---
        d4 = self.dec4(f, e4)  # 16x16, 512ch
        d3 = self.dec3(d4, e3)  # 32x32, 256ch
        d2 = self.dec2(d3, e2)  # 64x64, 128ch
        d1 = self.dec1(d2, e1)  # 128x128, 64ch

        # --- Output ---
        logits = self.final_conv(d1)

        if self.training and self.deep_supervision:
            aux_64 = self.aux_head_64(d2)
            aux_32 = self.aux_head_32(d3)
            # Return list for compound loss calculation
            return [logits, aux_64, aux_32]

        return logits
