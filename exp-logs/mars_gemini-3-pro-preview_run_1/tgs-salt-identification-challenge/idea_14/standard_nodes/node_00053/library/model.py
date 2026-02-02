import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Module.
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1), nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Squeeze & Excitation
        cse = self.cSE(x).unsqueeze(2).unsqueeze(3) * x
        # Spatial Squeeze & Excitation
        sse = self.sSE(x) * x
        # Concurrent combination
        return cse + sse


class ResidualUnit(nn.Module):
    """
    Standard Residual Block: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    (Pre-activation style can be used, but here we use standard ResNet V1 style
    or a robust variant for segmentation: Conv-BN-ReLU-Conv-BN-ReLU + Shortcut)

    Here we implement: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualUnit, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

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

        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder Block with Bilinear Upsampling, Skip Connection, Residual Unit, and SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # Bilinear upsampling is stateless, defined in forward or as a module
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Convolution to reduce channels after concatenation if necessary,
        # or just feed into the residual unit.
        # Input to ResidualUnit will be in_channels + skip_channels
        self.conv_block = ResidualUnit(
            in_channels + skip_channels, out_channels, stride=1
        )

        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)

        # Handle potential padding issues if dimensions don't match exactly due to odd input sizes
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(
                x,
                size=(skip.size(2), skip.size(3)),
                mode="bilinear",
                align_corners=True,
            )

        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        x = self.scse(x)
        return x


class DeepResUNet(nn.Module):
    """
    Deep Residual U-Net with Depth Fusion and Deep Supervision.
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(DeepResUNet, self).__init__()

        # Encoder
        # Input fusion: Image channels + 1 Depth channel
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels + 1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder Blocks (expanding to bottleneck of 512)
        # c1: 64 -> 64 (Resolution 128x128)
        self.enc1 = ResidualUnit(64, 64, stride=1)

        # c2: 64 -> 128 (Resolution 64x64)
        self.enc2 = ResidualUnit(64, 128, stride=2)

        # c3: 128 -> 256 (Resolution 32x32)
        self.enc3 = ResidualUnit(128, 256, stride=2)

        # Bottleneck: 256 -> 512 (Resolution 16x16)
        self.center = ResidualUnit(256, 512, stride=2)

        # Decoder
        # dec3: 512 (center) + 256 (enc3) -> 256 (Resolution 32x32)
        self.dec3 = DecoderBlock(512, 256, 256)

        # dec2: 256 (dec3) + 128 (enc2) -> 128 (Resolution 64x64)
        self.dec2 = DecoderBlock(256, 128, 128)

        # dec1: 128 (dec2) + 64 (enc1) -> 64 (Resolution 128x128)
        self.dec1 = DecoderBlock(128, 64, 64)

        # Deep Supervision Heads
        # Head at 32x32
        self.aux_head1 = nn.Conv2d(256, out_channels, kernel_size=1)

        # Head at 64x64
        self.aux_head2 = nn.Conv2d(128, out_channels, kernel_size=1)

        # Final Head at 128x128
        self.final_head = nn.Conv2d(64, out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, z):
        """
        Args:
            x: Image tensor (B, C, H, W)
            z: Depth tensor (B,) or (B, 1)
        """
        # 1. Depth Fusion
        # Normalize z to be roughly in 0-1 range (assuming z in feet/meters ~50-1000)
        z_norm = z.float() / 1000.0
        z_norm = z_norm.view(-1, 1, 1, 1)
        # Expand z to match image spatial dimensions
        z_map = z_norm.expand(-1, 1, x.size(2), x.size(3))

        x = torch.cat([x, z_map], dim=1)

        # 2. Encoder
        x = self.conv_in(x)  # 64, H, W

        e1 = self.enc1(x)  # 64, H, W
        e2 = self.enc2(e1)  # 128, H/2, W/2
        e3 = self.enc3(e2)  # 256, H/4, W/4
        c = self.center(e3)  # 512, H/8, W/8

        # 3. Decoder
        d3 = self.dec3(c, e3)  # 256, H/4, W/4
        d2 = self.dec2(d3, e2)  # 128, H/2, W/2
        d1 = self.dec1(d2, e1)  # 64, H, W

        # 4. Heads
        logits_32 = self.aux_head1(d3)
        logits_64 = self.aux_head2(d2)
        logits_128 = self.final_head(d1)

        # 5. Return
        if self.training:
            # Return list for deep supervision loss
            return [logits_128, logits_64, logits_32]
        else:
            # Return only final high-res logits for inference
            return logits_128
