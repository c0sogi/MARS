import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE) Module.
    Ref: "Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks"
    """

    def __init__(self, in_channels, reduction=16):
        super(SCSEModule, self).__init__()

        # Channel Squeeze & Excitation (cSE)
        # Global Average Pooling -> Dense -> ReLU -> Dense -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Squeeze & Excitation (sSE)
        # 1x1 Conv -> Sigmoid
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # cSE path: reweight channels
        # x: (B, C, H, W)
        # cse: (B, C) -> (B, C, 1, 1)
        batch, channels, _, _ = x.size()
        cse = self.cSE(x).view(batch, channels, 1, 1)
        u_cse = x * cse

        # sSE path: reweight spatial locations
        # sse: (B, 1, H, W)
        sse = self.sSE(x)
        u_sse = x * sse

        # Concurrent combination
        return u_cse + u_sse


class ResidualBlock(nn.Module):
    """
    Standard Residual Block:
    x -> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> + -> ReLU
         |                                          ^
         |__________________________________________|
         (with optional projection if channels/stride change)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

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

        self.downsample = None
        # If input shape/channels don't match output, project the shortcut
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
    Decoder Block with Attention:
    Upsample -> Concat -> ResidualBlock -> SCSE
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # We use bilinear upsampling in the forward pass, so no learnable layer here for upsampling
        # The input to the conv block will be in_channels (from below) + skip_channels (from encoder)

        self.res_block = ResidualBlock(in_channels + skip_channels, out_channels)
        self.attention = SCSEModule(out_channels)

    def forward(self, x, skip):
        # x: input from previous decoder layer (lower resolution)
        # skip: connection from encoder (higher resolution)

        # Upsample x to match skip size
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Refine features
        x = self.res_block(x)

        # Apply attention
        x = self.attention(x)

        return x


class DeepResUNet(nn.Module):
    """
    Deeply Supervised Residual U-Net.

    Encoder: 5 stages of Residual Blocks + MaxPool.
    Decoder: 4 stages of Attentive Decoder Blocks.
    Deep Supervision: Heads at 32x32, 64x64, 128x128.
    """

    def __init__(self, in_channels=2, out_channels=1, deep_supervision=True):
        super(DeepResUNet, self).__init__()

        self.deep_supervision = deep_supervision
        filters = Config.ENCODER_FILTERS  # [64, 128, 256, 512, 1024]

        # --- Encoder ---
        # Stage 1
        self.enc1 = ResidualBlock(in_channels, filters[0])
        self.pool1 = nn.MaxPool2d(2, 2)

        # Stage 2
        self.enc2 = ResidualBlock(filters[0], filters[1])
        self.pool2 = nn.MaxPool2d(2, 2)

        # Stage 3
        self.enc3 = ResidualBlock(filters[1], filters[2])
        self.pool3 = nn.MaxPool2d(2, 2)

        # Stage 4
        self.enc4 = ResidualBlock(filters[2], filters[3])
        self.pool4 = nn.MaxPool2d(2, 2)

        # Center (Stage 5)
        self.center = ResidualBlock(filters[3], filters[4])

        # --- Decoder ---
        # Dec 4 (16x16): Input 1024, Skip 512 -> Out 512
        self.dec4 = DecoderBlock(filters[4], filters[3], filters[3])

        # Dec 3 (32x32): Input 512, Skip 256 -> Out 256
        self.dec3 = DecoderBlock(filters[3], filters[2], filters[2])
        self.head3 = nn.Conv2d(filters[2], out_channels, kernel_size=1)

        # Dec 2 (64x64): Input 256, Skip 128 -> Out 128
        self.dec2 = DecoderBlock(filters[2], filters[1], filters[1])
        self.head2 = nn.Conv2d(filters[1], out_channels, kernel_size=1)

        # Dec 1 (128x128): Input 128, Skip 64 -> Out 64
        self.dec1 = DecoderBlock(filters[1], filters[0], filters[0])
        self.head1 = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 2, 128, 128)

        # Encoder
        e1 = self.enc1(x)  # (B, 64, 128, 128)
        p1 = self.pool1(e1)  # (B, 64, 64, 64)

        e2 = self.enc2(p1)  # (B, 128, 64, 64)
        p2 = self.pool2(e2)  # (B, 128, 32, 32)

        e3 = self.enc3(p2)  # (B, 256, 32, 32)
        p3 = self.pool3(e3)  # (B, 256, 16, 16)

        e4 = self.enc4(p3)  # (B, 512, 16, 16)
        p4 = self.pool4(e4)  # (B, 512, 8, 8)

        # Center
        c = self.center(p4)  # (B, 1024, 8, 8)

        # Decoder
        d4 = self.dec4(c, e4)  # (B, 512, 16, 16)

        d3 = self.dec3(d4, e3)  # (B, 256, 32, 32)
        out3 = self.head3(d3)  # DS Head 32x32

        d2 = self.dec2(d3, e2)  # (B, 128, 64, 64)
        out2 = self.head2(d2)  # DS Head 64x64

        d1 = self.dec1(d2, e1)  # (B, 64, 128, 128)
        out1 = self.head1(d1)  # Final Head 128x128

        if self.deep_supervision and self.training:
            # Return list of outputs for Deep Supervision Loss
            # Order: [High Res, Medium Res, Low Res]
            return [out1, out2, out3]
        else:
            # Return only the final high-resolution output
            return out1
