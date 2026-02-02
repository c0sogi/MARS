import torch
import torch.nn as nn
from library.config import Config


class DilatedResidualBlock(nn.Module):
    """
    A Residual Block that maintains spatial resolution using dilated convolutions.
    Structure:
        Input -> Conv3x3(d) -> BN -> ReLU -> Conv3x3(d) -> BN -> + -> ReLU
              |___________________________________________________|
    """

    def __init__(self, channels, dilation):
        super().__init__()
        # Padding must equal dilation to maintain spatial dimensions (H, W)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,  # Bias is redundant with BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class FRUNet(nn.Module):
    """
    Full-Resolution Dilated U-Net (FR-UNet).

    A symmetric encoder-decoder architecture that operates entirely at the
    original spatial resolution (no downsampling/pooling). It uses dilated
    convolutions to expand the receptive field and skip connections to
    fuse low-level texture with high-level context.
    """

    def __init__(self):
        super().__init__()

        # Load hyperparameters from Config
        in_channels = Config.IN_CHANNELS
        base_channels = Config.BASE_CHANNELS
        encoder_dilations = Config.ENCODER_DILATIONS
        decoder_dilations = Config.DECODER_DILATIONS

        # 1. Input Stem (Intensity Decoupling)
        # Projects Z-slices to feature space and normalizes intensity
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=1, bias=False),
            # affine=True allows learning scale/shift after removing instance mean/std
            nn.InstanceNorm2d(base_channels, affine=True),
        )

        # 2. Dilated Encoder (Context Expansion)
        # Progressively increases dilation to capture larger context
        self.encoder_blocks = nn.ModuleList()
        for d in encoder_dilations:
            self.encoder_blocks.append(DilatedResidualBlock(base_channels, dilation=d))

        # 3. Dilated Decoder (Hierarchical Refinement)
        # Progressively decreases dilation
        self.decoder_blocks = nn.ModuleList()
        for d in decoder_dilations:
            self.decoder_blocks.append(DilatedResidualBlock(base_channels, dilation=d))

        # 4. Classification Head
        self.head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (Batch, 65, H, W)

        # Stem
        x = self.stem(x)

        # Encoder Pass
        # We store outputs for skip connections
        encoder_outputs = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_outputs.append(x)

        # Decoder Pass
        # We iterate through decoder blocks and add the corresponding encoder output.
        # Encoder outputs are [e1, e2, e3, e4] (assuming 4 blocks).
        # Decoder blocks correspond to dilations matching the encoder in reverse order.
        # D1 (d=8) matches E4 (d=8)
        # D2 (d=4) matches E3 (d=4)
        # ...

        # Reverse encoder outputs to align with decoder sequence
        skips = encoder_outputs[::-1]

        for i, block in enumerate(self.decoder_blocks):
            # Summation Skip Connection
            # Input to decoder block = Previous Decoder Output + Corresponding Encoder Output
            x = x + skips[i]
            x = block(x)

        # Head
        x = self.head(x)

        return x
