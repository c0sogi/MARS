import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import STREAM_A_CONFIG, STREAM_B_CONFIG


class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Bilinear upsampling to increase spatial dimensions
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # 1x1 convolution to reduce channels of the upsampled feature map
        # to match the skip connection channels (out_channels).
        # This keeps the parameter count controlled.
        self.reduce = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # DoubleConv takes concatenated input (skip + upsampled)
        # Skip has `out_channels`, Upsampled (after reduce) has `out_channels`
        # Total input channels = 2 * out_channels
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x1, x2):
        # x1: input from previous layer (bottleneck or up)
        # x2: skip connection from encoder
        x1 = self.up(x1)
        x1 = self.reduce(x1)

        # Handle potential padding issues if dimensions are not perfectly divisible
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        if diffX != 0 or diffY != 0:
            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Modular U-Net architecture that constructs layers based on provided filter lists.
    """

    def __init__(
        self, input_channels, output_channels, encoder_filters, decoder_filters
    ):
        super(UNet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.encoder_filters = encoder_filters
        self.decoder_filters = decoder_filters

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Build Encoder (Down) Path ---
        in_ch = input_channels
        # Iterate through all encoder filters except the last one (which is for bottleneck)
        for out_ch in encoder_filters[:-1]:
            self.downs.append(DoubleConv(in_ch, out_ch))
            in_ch = out_ch

        # --- Build Bottleneck ---
        # Connects the last encoder block to the bottleneck
        self.bottleneck = DoubleConv(encoder_filters[-2], encoder_filters[-1])

        # --- Build Decoder (Up) Path ---
        # Input to first up block is the bottleneck output
        in_ch = encoder_filters[-1]
        for out_ch in decoder_filters:
            self.ups.append(Up(in_ch, out_ch))
            in_ch = out_ch

        # --- Final Classification Layer ---
        self.final_conv = nn.Conv2d(decoder_filters[-1], output_channels, kernel_size=1)

    def forward(self, x):
        skips = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        # Reverse skips to match up blocks (deepest skip first)
        skips = skips[::-1]

        for i, up in enumerate(self.ups):
            x = up(x, skips[i])

        return self.final_conv(x)


def get_context_specialist():
    """
    Constructs the Stream A: Context Specialist (4-Level U-Net)
    based on configuration.
    """
    cfg = STREAM_A_CONFIG
    return UNet(
        input_channels=cfg["input_channels"],
        output_channels=cfg["output_channels"],
        encoder_filters=cfg["encoder_filters"],
        decoder_filters=cfg["decoder_filters"],
    )


def get_texture_specialist():
    """
    Constructs the Stream B: Texture Specialist (3-Level U-Net)
    based on configuration.
    """
    cfg = STREAM_B_CONFIG
    return UNet(
        input_channels=cfg["input_channels"],
        output_channels=cfg["output_channels"],
        encoder_filters=cfg["encoder_filters"],
        decoder_filters=cfg["decoder_filters"],
    )
