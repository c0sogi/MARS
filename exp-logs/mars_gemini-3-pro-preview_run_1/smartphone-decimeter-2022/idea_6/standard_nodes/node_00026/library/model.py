import torch
import torch.nn as nn
import torch.nn.functional as F
import library.config as C


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional dropout.
    Structure: Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> Add(Input) -> ReLU
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut connection to match dimensions if necessary
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = self.act(x)
        return x


class TransformerBottleneck(nn.Module):
    """
    Transformer Bottleneck with Device Embedding and Positional Encoding.
    Operates on the compressed latent representation.
    """

    def __init__(
        self,
        channels,
        num_layers,
        num_heads,
        ff_dim,
        num_devices,
        dropout=0.1,
        max_len=10000,
    ):
        super().__init__()
        self.channels = channels

        # Device Context Embedding
        self.device_emb = nn.Embedding(num_devices, channels)

        # Learnable Positional Encoding
        # Initialized with normal distribution
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, channels))
        nn.init.normal_(self.pos_emb, mean=0, std=0.02)

        # Transformer Encoder
        # batch_first=True ensures input is (Batch, Length, Channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, phone_idx):
        """
        Args:
            x: Feature map from encoder (Batch, Channels, Length)
            phone_idx: Device indices (Batch,)
        """
        B, C, L = x.shape

        # Permute to (Batch, Length, Channels) for Transformer
        x = x.permute(0, 2, 1)

        # Add Device Embedding (broadcast across time)
        dev_emb = self.device_emb(phone_idx).unsqueeze(1)  # (B, 1, C)
        x = x + dev_emb

        # Add Positional Encoding
        # Dynamically interpolate if sequence is longer than max_len (rare but safe)
        if L <= self.pos_emb.shape[1]:
            x = x + self.pos_emb[:, :L, :]
        else:
            pos_emb_resized = F.interpolate(
                self.pos_emb.permute(0, 2, 1),
                size=L,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
            x = x + pos_emb_resized

        # Apply Transformer
        x = self.transformer(x)

        # Permute back to (Batch, Channels, Length) for Decoder
        x = x.permute(0, 2, 1)

        return x


class TransResUNet(nn.Module):
    """
    Context-Aware 1D TransResUNet.
    Combines ResNet Encoder/Decoder with a Transformer Bottleneck.
    """

    def __init__(self):
        super().__init__()

        self.input_dim = C.INPUT_DIM
        self.output_dim = C.OUTPUT_DIM

        # Configuration from config.py
        enc_channels = C.ENCODER_CHANNELS  # e.g., [32, 64, 128, 256]

        # --- Encoder ---
        # Initial projection (Stem)
        self.stem = nn.Sequential(
            nn.Conv1d(
                self.input_dim, enc_channels[0], kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm1d(enc_channels[0]),
            nn.ReLU(inplace=True),
        )

        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        in_ch = enc_channels[0]
        for out_ch in enc_channels:
            self.enc_blocks.append(
                ResBlock1D(in_ch, out_ch, kernel_size=C.KERNEL_SIZE, dropout=C.DROPOUT)
            )
            self.downsamples.append(nn.MaxPool1d(2))
            in_ch = out_ch

        # --- Bottleneck ---
        self.bottleneck = TransformerBottleneck(
            channels=enc_channels[-1],
            num_layers=C.TRANSFORMER_LAYERS,
            num_heads=C.TRANSFORMER_HEADS,
            ff_dim=C.TRANSFORMER_FF_DIM,
            num_devices=C.NUM_DEVICES,
            dropout=C.DROPOUT,
        )

        # --- Decoder ---
        self.up_samples = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        # Reverse channels for decoder construction
        # Example: [256, 128, 64, 32]
        rev_channels = enc_channels[::-1]

        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i]
            skip_ch = rev_channels[
                i
            ]  # Skip connection has same dim as input from below
            out_ch = rev_channels[i + 1]

            self.up_samples.append(
                nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
            )
            # Input to block is concat of Upsampled + Skip
            self.dec_blocks.append(
                ResBlock1D(
                    in_ch + skip_ch,
                    out_ch,
                    kernel_size=C.KERNEL_SIZE,
                    dropout=C.DROPOUT,
                )
            )

        # Final Upsample block to return to original resolution
        self.final_up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        # Skip connection from Stem/Enc[0] is enc_channels[0]
        self.final_dec_block = ResBlock1D(
            rev_channels[-1] + enc_channels[0],
            enc_channels[0],
            kernel_size=C.KERNEL_SIZE,
            dropout=C.DROPOUT,
        )

        # --- Output Head ---
        self.head = nn.Conv1d(enc_channels[0], self.output_dim, kernel_size=1)

    def forward(self, x, phone_idx):
        """
        Args:
            x: (Batch, InputDim, Length)
            phone_idx: (Batch,)
        Returns:
            out: (Batch, OutputDim, Length)
        """
        B, C, L = x.shape

        # 1. Pad input to be divisible by 16 (2^4 downsampling)
        pad_factor = 16
        pad_len = (pad_factor - (L % pad_factor)) % pad_factor
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        # 2. Stem
        x = self.stem(x)

        # 3. Encoder Path
        skips = []
        # We need the output of the block BEFORE pooling for the skip connection
        # But the first block input is the stem output.
        # Let's trace:
        # Stem -> x (32)
        # Block0(32->32) -> x_enc0. Skip=x_enc0. Pool -> x_down0
        # Block1(32->64) -> x_enc1. Skip=x_enc1. Pool -> x_down1
        # ...

        for block, down in zip(self.enc_blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = down(x)

        # 4. Bottleneck
        x = self.bottleneck(x, phone_idx)

        # 5. Decoder Path
        # Skips are [Enc0, Enc1, Enc2, Enc3] (Low level -> High level)
        # Decoder needs High level -> Low level
        skips = skips[::-1]

        # Iterate through decoder layers (excluding final restoration)
        for i, (up, block) in enumerate(zip(self.up_samples, self.dec_blocks)):
            x = up(x)
            # Concatenate with corresponding skip connection
            # skips[0] corresponds to the input of the bottleneck (before last pool)
            skip = skips[i]
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # 6. Final Restoration to Original Resolution
        x = self.final_up(x)
        # Concatenate with the lowest level skip (Enc0)
        skip = skips[-1]
        x = torch.cat([x, skip], dim=1)
        x = self.final_dec_block(x)

        # 7. Output Head
        out = self.head(x)

        # 8. Remove Padding
        if pad_len > 0:
            out = out[:, :, :-pad_len]

        return out
