import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions and a skip connection.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
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
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        # Projection shortcut if dimensions change
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
    Decoder block with Bilinear Upsampling, Concatenation, Conv Block, and SCSE.
    """

    def __init__(self, in_channels, skip_channels, out_channels, use_scse=True):
        super().__init__()
        # Input channels = upsampled_channels + skip_channels
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.use_scse = use_scse
        if use_scse:
            self.scse = SCSEBlock(out_channels)

    def forward(self, x, skip):
        # 1. Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # 2. Concatenate with Skip Connection
        if skip is not None:
            # Handle slight dimension mismatches due to padding/pooling
            if x.size() != skip.size():
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)

        # 3. Convolutional Block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 4. Attention
        if self.use_scse:
            x = self.scse(x)
        return x


class HyperResUNet(nn.Module):
    """
    Hypercolumn Deep Residual U-Net.
    Aggregates features from all decoder levels for the final prediction.
    """

    def __init__(self):
        super().__init__()

        # ---------------------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------------------
        filters = Config.ENCODER_FILTERS
        in_ch = Config.INPUT_CHANNELS
        self.use_hypercolumns = Config.USE_HYPERCOLUMNS
        self.deep_supervision = Config.DEEP_SUPERVISION
        use_scse = Config.USE_SCSE

        # ---------------------------------------------------------------------
        # Encoder (Custom Residual Backbone)
        # ---------------------------------------------------------------------
        # Layer 1: Input -> 64
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            ResidualBlock(filters, filters),
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Layer 2: 64 -> 128
        self.enc2 = ResidualBlock(filters, filters * 2)

        # Layer 3: 128 -> 256
        self.enc3 = ResidualBlock(filters * 2, filters * 4)

        # Layer 4: 256 -> 512
        self.enc4 = ResidualBlock(filters * 4, filters * 8)

        # Center (Bridge): 512 -> 1024
        self.center = ResidualBlock(filters * 8, filters * 16)

        # ---------------------------------------------------------------------
        # Decoder
        # ---------------------------------------------------------------------
        # Decoder 4: Center(1024) + Enc4(512) -> 512
        self.dec4 = DecoderBlock(filters * 16, filters * 8, filters * 8, use_scse)

        # Decoder 3: Dec4(512) + Enc3(256) -> 256
        self.dec3 = DecoderBlock(filters * 8, filters * 4, filters * 4, use_scse)

        # Decoder 2: Dec3(256) + Enc2(128) -> 128
        self.dec2 = DecoderBlock(filters * 4, filters * 2, filters * 2, use_scse)

        # Decoder 1: Dec2(128) + Enc1(64) -> 64
        self.dec1 = DecoderBlock(filters * 2, filters, filters, use_scse)

        # ---------------------------------------------------------------------
        # Heads (Hypercolumns & Deep Supervision)
        # ---------------------------------------------------------------------
        if self.use_hypercolumns:
            # Aggregate channels from all decoder levels
            # 512 + 256 + 128 + 64 = 960 (if filters=64)
            hyper_channels = (filters * 8) + (filters * 4) + (filters * 2) + filters
            self.final_conv = nn.Conv2d(hyper_channels, 1, kernel_size=1)
        else:
            self.final_conv = nn.Conv2d(filters, 1, kernel_size=1)

        if self.deep_supervision:
            # Auxiliary heads for intermediate layers
            self.aux_head4 = nn.Conv2d(filters * 8, 1, kernel_size=1)
            self.aux_head3 = nn.Conv2d(filters * 4, 1, kernel_size=1)
            self.aux_head2 = nn.Conv2d(filters * 2, 1, kernel_size=1)

            # If using hypercolumns, the 'main' head is the aggregate,
            # so Dec1 is also an intermediate representation we can supervise.
            if self.use_hypercolumns:
                self.aux_head1 = nn.Conv2d(filters, 1, kernel_size=1)

    def forward(self, x):
        # ---------------------------------------------------------------------
        # Encoder Forward
        # ---------------------------------------------------------------------
        e1 = self.enc1(x)  # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))  # (B, 512, H/8, W/8)
        c = self.center(self.pool(e4))  # (B, 1024, H/16, W/16)

        # ---------------------------------------------------------------------
        # Decoder Forward
        # ---------------------------------------------------------------------
        d4 = self.dec4(c, e4)  # (B, 512, H/8, W/8)
        d3 = self.dec3(d4, e3)  # (B, 256, H/4, W/4)
        d2 = self.dec2(d3, e2)  # (B, 128, H/2, W/2)
        d1 = self.dec1(d2, e1)  # (B, 64, H, W)

        outputs = []

        # ---------------------------------------------------------------------
        # Hypercolumn Aggregation
        # ---------------------------------------------------------------------
        if self.use_hypercolumns:
            # Upsample all decoder outputs to input resolution
            target_size = d1.shape[2:]
            d4_up = F.interpolate(
                d4, size=target_size, mode="bilinear", align_corners=True
            )
            d3_up = F.interpolate(
                d3, size=target_size, mode="bilinear", align_corners=True
            )
            d2_up = F.interpolate(
                d2, size=target_size, mode="bilinear", align_corners=True
            )

            # Concatenate along channel dimension
            hyper = torch.cat([d4_up, d3_up, d2_up, d1], dim=1)

            # Final prediction
            logits = self.final_conv(hyper)
            outputs.append(logits)
        else:
            logits = self.final_conv(d1)
            outputs.append(logits)

        # ---------------------------------------------------------------------
        # Deep Supervision (Training Only)
        # ---------------------------------------------------------------------
        if self.deep_supervision and self.training:
            target_size = d1.shape[2:]

            # Aux 4
            a4 = self.aux_head4(d4)
            a4 = F.interpolate(
                a4, size=target_size, mode="bilinear", align_corners=True
            )
            outputs.append(a4)

            # Aux 3
            a3 = self.aux_head3(d3)
            a3 = F.interpolate(
                a3, size=target_size, mode="bilinear", align_corners=True
            )
            outputs.append(a3)

            # Aux 2
            a2 = self.aux_head2(d2)
            a2 = F.interpolate(
                a2, size=target_size, mode="bilinear", align_corners=True
            )
            outputs.append(a2)

            if self.use_hypercolumns:
                a1 = self.aux_head1(d1)
                # d1 is already at target size
                outputs.append(a1)

        # Return list for DeepSupervisionLoss during training, or single tensor for inference
        if self.training:
            return outputs
        else:
            return outputs[0]
