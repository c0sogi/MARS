import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class ResidualBlock1D(nn.Module):
    """
    1D Residual Block with two convolution layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, 1, padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
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


class ConvEncoder(nn.Module):
    """
    Encoder composed of stacked ResidualBlock1D layers and max pooling.
    """

    def __init__(self, channels):
        super(ConvEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # channels example: [10, 32, 64, 128]
        # Create blocks: 10->32, 32->64, 64->128
        for i in range(len(channels) - 1):
            self.blocks.append(ResidualBlock1D(channels[i], channels[i + 1]))

    def forward(self, x):
        skips = []
        for block in self.blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)
        return skips, x


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for the Transformer.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # Shape: (L, 1, D)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (L, B, D)
        return x + self.pe[: x.size(0), :]


class TransformerBottleneck(nn.Module):
    """
    Transformer Encoder bottleneck to capture global context.
    """

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerBottleneck, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model

    def forward(self, x):
        # Input x: (B, C, L)
        # Permute to (L, B, C) for Transformer
        x = x.permute(2, 0, 1)
        x = x * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Permute back to (B, C, L)
        x = x.permute(1, 2, 0)
        return x


class ConvDecoder(nn.Module):
    """
    Decoder composed of upsampling and ResidualBlock1D layers, fusing skip connections.
    """

    def __init__(self, encoder_channels, decoder_channels):
        super(ConvDecoder, self).__init__()
        self.blocks = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)

        # encoder_channels: [10, 32, 64, 128]
        # decoder_channels: [128, 64, 32, 16]

        # We need to match skips from encoder.
        # Skips are: [32, 64, 128] (indices 0, 1, 2 of output list)
        # Bottleneck output is 128.

        # Block 1: In=128(bottleneck), Skip=128(enc_ch[-1]), Out=128(dec_ch[0])
        self.blocks.append(
            ResidualBlock1D(128 + encoder_channels[-1], decoder_channels[0])
        )

        # Block 2: In=128(prev), Skip=64(enc_ch[-2]), Out=64(dec_ch[1])
        self.blocks.append(
            ResidualBlock1D(
                decoder_channels[0] + encoder_channels[-2], decoder_channels[1]
            )
        )

        # Block 3: In=64(prev), Skip=32(enc_ch[-3]), Out=32(dec_ch[2])
        self.blocks.append(
            ResidualBlock1D(
                decoder_channels[1] + encoder_channels[-3], decoder_channels[2]
            )
        )

        # Final conv block (no upsample/skip matching, just processing)
        # In=32, Out=16
        self.final_block = ResidualBlock1D(decoder_channels[2], decoder_channels[3])

    def forward(self, x, skips):
        # x is bottleneck output
        # skips is list [skip_l1, skip_l2, skip_l3]

        # Reverse skips to match decoder order (deepest skip first)
        skips = skips[::-1]

        for i, block in enumerate(self.blocks):
            x = self.upsample(x)

            # Handle potential size mismatch due to odd input lengths
            skip = skips[i]
            if x.size(2) != skip.size(2):
                x = F.interpolate(
                    x, size=skip.size(2), mode="linear", align_corners=True
                )

            x = torch.cat([x, skip], dim=1)
            x = block(x)

        x = self.final_block(x)
        return x


class TransUNet1D(nn.Module):
    """
    Hybrid 1D TransUNet model for sequence-to-sequence regression.
    """

    def __init__(self):
        super(TransUNet1D, self).__init__()

        self.encoder = ConvEncoder(Config.ENCODER_CHANNELS)

        self.bottleneck = TransformerBottleneck(
            d_model=Config.TRANSFORMER_EMBED_DIM,
            nhead=Config.TRANSFORMER_NUM_HEADS,
            num_layers=Config.TRANSFORMER_NUM_LAYERS,
            dim_feedforward=Config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=Config.TRANSFORMER_DROPOUT,
        )

        self.decoder = ConvDecoder(Config.ENCODER_CHANNELS, Config.DECODER_CHANNELS)

        self.head = nn.Conv1d(
            Config.DECODER_CHANNELS[-1], Config.OUTPUT_DIM, kernel_size=1
        )

    def forward(self, x):
        # x: (B, C, L)

        # Pad input to be divisible by 8 (2^3 pooling layers)
        # This ensures shapes match in decoder
        original_len = x.size(2)
        pad_len = (8 - (original_len % 8)) % 8
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        skips, encoded = self.encoder(x)

        bottleneck = self.bottleneck(encoded)

        decoded = self.decoder(bottleneck, skips)

        output = self.head(decoded)

        # Crop back to original length
        if pad_len > 0:
            output = output[:, :, :original_len]

        return output
