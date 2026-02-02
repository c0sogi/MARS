import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class StochasticDepth(nn.Module):
    """
    Stochastic Depth (DropPath) regularization.
    Randomly drops residual paths during training to regularize the network.
    """

    def __init__(self, prob=0.0):
        super().__init__()
        self.prob = prob

    def forward(self, x):
        if not self.training or self.prob == 0.0:
            return x

        # Calculate keep probability
        keep_prob = 1.0 - self.prob

        # Compute shape for broadcasting: (N, 1, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        # Generate Bernoulli mask
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        random_tensor = random_tensor.to(x.dtype)

        # Scale output to maintain expected value
        output = x.div(keep_prob) * random_tensor
        return output


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation (scSE).
    Recalibrates feature maps spatially and channel-wise.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel SE: Global Avg Pool -> Dense -> ReLU -> Dense -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial SE: Conv 1x1 -> Sigmoid
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        # Concurrent excitation: Additive combination
        return x * self.cSE(x) + x * self.sSE(x)


class ResidualBlock(nn.Module):
    """
    ResNet-style Residual Block with Stochastic Depth.
    """

    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super().__init__()
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
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection handling for stride or channel changes
        self.downsample = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.drop_path = StochasticDepth(drop_prob)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)

        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # Apply stochastic depth to the residual branch
        residual = self.drop_path(residual)

        out = self.downsample(x) + residual
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    """
    Decoder block combining Bilinear Upsampling, Skip Connection, Convolutions, and SCSE Attention.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
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

        self.scse = SCSEModule(out_channels)

    def forward(self, x, skip):
        # Bilinear Upsampling
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Handle potential dimension mismatch due to padding/cropping
        if x.size() != skip.size():
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Apply Attention
        x = self.scse(x)
        return x


class DeepResUNet(nn.Module):
    """
    High-Capacity Deep Residual U-Net.
    Features:
    - Input Depth Fusion
    - ResNet Encoder (up to 1024 filters)
    - Stochastic Depth Regularization
    - SCSE Attention Decoder
    - Deep Supervision Heads
    """

    def __init__(self):
        super().__init__()

        # Load Hyperparameters
        start_filters = Config.ENCODER_FILTERS  # 64
        center_filters = Config.CENTER_FILTERS  # 1024
        use_stochastic = Config.USE_STOCHASTIC_DEPTH
        max_drop_rate = Config.DROP_PATH_RATE if use_stochastic else 0.0
        self.deep_supervision = Config.DEEP_SUPERVISION

        # Input Fusion: Image (1 ch) + Depth (1 ch) = 2 channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(2, start_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(start_filters),
            nn.ReLU(inplace=True),
        )

        # Calculate drop rates linearly increasing with depth
        # Stages: Enc1, Enc2, Enc3, Center (4 stages with drop path)
        dpr = [x.item() for x in torch.linspace(0, max_drop_rate, 4)]

        # Encoder (ResNet-style)
        # Input: (B, 64, 128, 128)
        self.enc1 = ResidualBlock(
            start_filters, 128, stride=2, drop_prob=dpr[0]
        )  # -> (B, 128, 64, 64)
        self.enc2 = ResidualBlock(
            128, 256, stride=2, drop_prob=dpr[1]
        )  # -> (B, 256, 32, 32)
        self.enc3 = ResidualBlock(
            256, 512, stride=2, drop_prob=dpr[2]
        )  # -> (B, 512, 16, 16)

        # Center Bottleneck
        self.center = ResidualBlock(
            512, center_filters, stride=2, drop_prob=dpr[3]
        )  # -> (B, 1024, 8, 8)

        # Decoder
        # dec4: 1024 (center) + 512 (enc3) -> 512
        self.dec4 = DecoderBlock(center_filters, 512, 512)  # -> (B, 512, 16, 16)

        # dec3: 512 (dec4) + 256 (enc2) -> 256
        self.dec3 = DecoderBlock(512, 256, 256)  # -> (B, 256, 32, 32)

        # dec2: 256 (dec3) + 128 (enc1) -> 128
        self.dec2 = DecoderBlock(256, 128, 128)  # -> (B, 128, 64, 64)

        # dec1: 128 (dec2) + 64 (input_conv) -> 64
        self.dec1 = DecoderBlock(128, start_filters, 64)  # -> (B, 64, 128, 128)

        # Final Prediction Head
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

        # Auxiliary Heads for Deep Supervision
        if self.deep_supervision:
            self.aux_head1 = nn.Conv2d(256, 1, kernel_size=1)  # From dec3 (32x32)
            self.aux_head2 = nn.Conv2d(128, 1, kernel_size=1)  # From dec2 (64x64)

    def forward(self, x, depth):
        """
        Args:
            x: (B, 1, H, W) Input Image
            depth: (B,) or (B, 1) Depth values
        """
        # 1. Depth Fusion
        b, c, h, w = x.shape
        # Normalize depth roughly to 0-1 range (assuming max depth ~1000)
        depth_norm = depth.float() / 1000.0
        # Expand scalar depth to dense channel (B, 1, H, W)
        depth_channel = depth_norm.view(b, 1, 1, 1).expand(b, 1, h, w)
        # Concatenate: Input becomes 2 channels
        x = torch.cat([x, depth_channel], dim=1)

        # 2. Encoder Forward Pass
        x0 = self.input_conv(x)  # 128
        x1 = self.enc1(x0)  # 64
        x2 = self.enc2(x1)  # 32
        x3 = self.enc3(x2)  # 16

        # 3. Center
        c = self.center(x3)  # 8

        # 4. Decoder Forward Pass
        d4 = self.dec4(c, x3)  # 16
        d3 = self.dec3(d4, x2)  # 32
        d2 = self.dec2(d3, x1)  # 64
        d1 = self.dec1(d2, x0)  # 128

        # 5. Prediction
        logits = self.final_conv(d1)

        # 6. Deep Supervision Returns
        if self.deep_supervision and self.training:
            # Upsample aux predictions to target size for loss calculation
            aux1 = self.aux_head1(d3)
            aux1 = F.interpolate(aux1, size=(h, w), mode="bilinear", align_corners=True)

            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=(h, w), mode="bilinear", align_corners=True)

            return logits, aux2, aux1

        return logits
