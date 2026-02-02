import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with Batch Normalization and ReLU.
    Structure: Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+ Skip) -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResidualBlock1D, self).__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection handling
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
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


class ResUNet1D(nn.Module):
    """
    1D Residual U-Net for sequence-to-sequence regression.

    Architecture:
    - Encoder: 4 levels of Residual Blocks with MaxPooling.
    - Bottleneck: Dilated Residual Blocks for larger receptive field.
    - Decoder: 4 levels of Upsampling + Concat + Residual Blocks.
    - Head: Projection to output dimension (2: East, North).
    """

    def __init__(self, in_channels, out_channels=2, base_channels=64):
        super(ResUNet1D, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            ResidualBlock1D(base_channels, base_channels),
        )
        self.pool1 = nn.MaxPool1d(2)

        self.enc2 = ResidualBlock1D(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.enc3 = ResidualBlock1D(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.enc4 = ResidualBlock1D(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck (Dilated convolutions to capture global context)
        self.bottleneck = nn.Sequential(
            ResidualBlock1D(base_channels * 8, base_channels * 16, dilation=2),
            ResidualBlock1D(base_channels * 16, base_channels * 16, dilation=4),
        )

        # Decoder
        self.up4 = nn.ConvTranspose1d(
            base_channels * 16, base_channels * 8, kernel_size=2, stride=2
        )
        self.dec4 = ResidualBlock1D(
            base_channels * 16, base_channels * 8
        )  # Input is cat(up4, enc4)

        self.up3 = nn.ConvTranspose1d(
            base_channels * 8, base_channels * 4, kernel_size=2, stride=2
        )
        self.dec3 = ResidualBlock1D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose1d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.dec2 = ResidualBlock1D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose1d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.dec1 = ResidualBlock1D(base_channels * 2, base_channels)

        # Output Head
        self.head = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: (Batch, SeqLen, Features)
        # Permute to (Batch, Features, SeqLen) for Conv1d
        x = x.permute(0, 2, 1)

        # Pad input to be divisible by 16 (2^4 pooling layers)
        original_length = x.size(2)
        divisor = 16
        pad_len = (divisor - (original_length % divisor)) % divisor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b)
        # Handle potential size mismatch due to odd dimensions in pooling (though padding above helps)
        if d4.size(2) != e4.size(2):
            d4 = F.interpolate(d4, size=e4.size(2), mode="linear", align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        if d3.size(2) != e3.size(2):
            d3 = F.interpolate(d3, size=e3.size(2), mode="linear", align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        if d2.size(2) != e2.size(2):
            d2 = F.interpolate(d2, size=e2.size(2), mode="linear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.size(2) != e1.size(2):
            d1 = F.interpolate(d1, size=e1.size(2), mode="linear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Head
        out = self.head(d1)

        # Remove padding
        if pad_len > 0:
            out = out[:, :, :-pad_len]

        # Permute back to (Batch, SeqLen, OutChannels)
        out = out.permute(0, 2, 1)

        return out
