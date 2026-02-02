import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from library.config import Config


class ResidualBlock1d(nn.Module):
    """
    1D Residual Block with optional channel projection.
    Structure: Conv1d -> BN -> ReLU -> Conv1d -> BN -> (+ Residual) -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock1d, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
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


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for 1D sequences.
    """

    def __init__(self, d_model, max_len=100000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Shape: (1, max_len, d_model) for broadcasting
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        # Slice pe to the current sequence length
        if x.size(1) > self.pe.size(1):
            raise RuntimeError(
                f"Input sequence length {x.size(1)} exceeds PositionalEncoding max_len {self.pe.size(1)}"
            )
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransUNet1D(nn.Module):
    """
    Hybrid 1D U-Net with ResNet Encoder/Decoder and Transformer Bottleneck.
    """

    def __init__(self):
        super(TransUNet1D, self).__init__()

        self.input_dim = Config.INPUT_DIM
        self.output_dim = Config.OUTPUT_DIM

        encoder_dims = Config.ENCODER_CHANNELS  # e.g., [32, 64, 128, 256]
        decoder_dims = Config.DECODER_CHANNELS  # e.g., [128, 64, 32]

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Initial projection
        curr_dim = self.input_dim

        # Create encoder stages
        for out_dim in encoder_dims:
            # We use stride=1 in block and separate pooling layer for U-Net structure
            block = ResidualBlock1d(curr_dim, out_dim)
            self.encoder_blocks.append(block)
            curr_dim = out_dim

        # --- Transformer Bottleneck ---
        self.d_model = encoder_dims[-1]
        # Ensure max_len covers long GNSS tracks (24h+ at 1Hz)
        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=100000, dropout=Config.TRANSFORMER_DROPOUT
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=Config.TRANSFORMER_NHEAD,
            dim_feedforward=Config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=Config.TRANSFORMER_DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=Config.TRANSFORMER_NUM_LAYERS
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        # Reverse encoder dims for skip connections (excluding the last one which is the bottleneck)
        skip_dims = encoder_dims[:-1][::-1]  # e.g., [128, 64, 32]

        curr_dim = self.d_model

        for i, out_dim in enumerate(decoder_dims):
            skip_dim = skip_dims[i]

            # Upsampling layer (1x1 conv after interpolation usually, or just reduce channels)
            # Here we just define the block that takes concatenated input
            # Input to block: curr_dim (from upsample) + skip_dim

            # We will use interpolation for upsampling, then a 1x1 conv to reduce channel count if needed,
            # or just feed into ResBlock.
            # Let's use a ResBlock that reduces channels: (curr_dim + skip_dim) -> out_dim

            block = ResidualBlock1d(curr_dim + skip_dim, out_dim)
            self.decoder_blocks.append(block)
            curr_dim = out_dim

        # --- Head ---
        self.head = nn.Conv1d(curr_dim, self.output_dim, kernel_size=1)

    def forward(self, x, mask=None):
        """
        Args:
            x: (Batch, Channels, Length)
            mask: (Batch, Length) - Boolean mask indicating valid data (not padding).
                  Used for Transformer masking if needed (though usually pad masking is sufficient).
        """
        skips = []

        # --- Encoder Pass ---
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)

            # Store skip connection for all but the last block (bottleneck input)
            if i < len(self.encoder_blocks) - 1:
                skips.append(x)
                x = self.pool(x)

        # x is now at bottleneck resolution

        # --- Transformer Pass ---
        # Prepare for Transformer: (B, C, L) -> (B, L, C)
        b, c, l = x.shape
        x_trans = x.permute(0, 2, 1)

        # Add Positional Encoding
        x_trans = self.pos_encoder(x_trans)

        # Create padding mask for Transformer if mask is provided
        # Mask needs to be downsampled to current resolution
        src_key_padding_mask = None
        if mask is not None:
            # Downsample mask: simple nearest neighbor or max pooling logic
            # mask is (B, Original_L). We need (B, l)
            # We can interpolate the boolean mask as float then threshold back
            mask_float = mask.float().unsqueeze(1)  # (B, 1, Orig_L)
            mask_down = F.interpolate(mask_float, size=l, mode="nearest").squeeze(1)
            # Transformer expects True for padding (positions to ignore)
            # Our mask is True for valid. So invert.
            src_key_padding_mask = mask_down < 0.5

        # Apply Transformer
        x_trans = self.transformer(x_trans, src_key_padding_mask=src_key_padding_mask)

        # Reshape back: (B, L, C) -> (B, C, L)
        x = x_trans.permute(0, 2, 1)

        # --- Decoder Pass ---
        # skips list is [Block1_out, Block2_out, Block3_out]
        # We iterate backwards
        for i, block in enumerate(self.decoder_blocks):
            skip = skips.pop()

            # Upsample x to match skip length
            # Use interpolate to handle potential odd dimensions
            x = F.interpolate(x, size=skip.shape[2], mode="linear", align_corners=False)

            # Concatenate
            x = torch.cat([x, skip], dim=1)

            # Pass through decoder block
            x = block(x)

        # --- Output Head ---
        # x is now (B, 32, L_original/2) if we did one last upsample?
        # Wait, the last block in encoder was NOT pooled.
        # Let's trace:
        # Input (L) -> B1(L) -> Pool(L/2) -> B2(L/2) -> Pool(L/4) -> B3(L/4) -> Pool(L/8) -> B4(L/8)
        # Skips: [B1(L), B2(L/2), B3(L/4)]
        # Bottleneck: B4(L/8)
        # Decoder 1: Up(L/8->L/4) + Skip B3(L/4) -> Out(L/4)
        # Decoder 2: Up(L/4->L/2) + Skip B2(L/2) -> Out(L/2)
        # Decoder 3: Up(L/2->L)   + Skip B1(L)   -> Out(L)
        # Head: Out(L) -> Final(L)

        out = self.head(x)

        return out
