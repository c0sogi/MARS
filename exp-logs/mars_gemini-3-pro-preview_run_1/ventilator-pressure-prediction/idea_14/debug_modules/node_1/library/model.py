import torch
import torch.nn as nn
from library.config import Config


class MultiScaleCNN(nn.Module):
    """
    Multi-Scale 1D Convolutional Stem (Inception-style).
    Captures both fine-grained signal noise and smoothed trend derivatives
    using parallel convolutions with different kernel sizes.
    """

    def __init__(self, input_dim, hidden_dim, kernel_sizes):
        super().__init__()
        # Calculate intermediate channel dimension
        # We split the hidden dimension across the different kernels
        inter_dim = hidden_dim // len(kernel_sizes)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=inter_dim,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        # Projection to ensure exact hidden_dim size after concatenation
        concat_dim = inter_dim * len(kernel_sizes)
        self.proj = nn.Conv1d(concat_dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        # Input x: (Batch, Time, Features)
        # Conv1d expects: (Batch, Channels, Time)
        x = x.transpose(1, 2)

        # Apply parallel convolutions
        outs = [conv(x) for conv in self.convs]
        x = torch.cat(outs, dim=1)

        # Project and Activate
        x = self.proj(x)
        x = self.act(x)

        # Return to (Batch, Time, Hidden)
        return x.transpose(1, 2)


class CompositeBlock(nn.Module):
    """
    High-Capacity Composite Block.
    Consists of two sub-modules:
    1. Context-Injected Temporal Mixer (Bi-LSTM) with Physics Injection.
    2. Pointwise Channel Mixer (FFN).

    Crucially, this block uses pure additive residual connections WITHOUT
    Layer Normalization to preserve the absolute magnitude of the pressure signal.
    """

    def __init__(self, hidden_dim, physics_dim, dropout=0.1):
        super().__init__()

        # 1. Context-Injected Temporal Mixer
        # Input is concatenation of the residual stream and the static physics features.
        # This re-injects physical constraints (R, C, etc.) at every depth.
        self.lstm = nn.LSTM(
            input_size=hidden_dim + physics_dim,
            hidden_size=hidden_dim // 2,  # Bidirectional, so output will be hidden_dim
            bidirectional=True,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        # 2. Pointwise Channel Mixer
        # Standard FFN to mix features across channels independent of time
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, physics_feats):
        # x: (Batch, Time, Hidden)
        # physics_feats: (Batch, Time, Physics_Dim)

        # --- Temporal Mixing ---
        # Concatenate physics features to the input of the LSTM
        lstm_in = torch.cat([x, physics_feats], dim=-1)

        # LSTM output: (Batch, Time, Hidden_Dim)
        lstm_out, _ = self.lstm(lstm_in)

        # Additive Residual (Pure Add, no Norm)
        x = x + self.dropout1(lstm_out)

        # --- Channel Mixing ---
        ffn_out = self.ffn(x)

        # Additive Residual (Pure Add, no Norm)
        x = x + self.dropout2(ffn_out)

        return x


class HighCapacityCompositeModel(nn.Module):
    """
    High-Capacity Unnormalized Physics-Injected Composite CNN-LSTM-FFN.

    Architecture:
    - Multi-Scale CNN Stem
    - Stack of CompositeBlocks (Bi-LSTM + FFN) with Physics Injection
    - Deep Supervision (Auxiliary Head)
    - Final Linear Projection
    """

    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # Identify indices of physics features in the input tensor for injection
        try:
            self.physics_indices = [
                config.features.index(f) for f in config.physics_features
            ]
        except ValueError as e:
            raise ValueError(f"Physics feature configuration error: {e}")

        physics_dim = len(self.physics_indices)

        # Stem
        self.stem = MultiScaleCNN(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            kernel_sizes=config.stem_kernel_sizes,
        )

        # Backbone
        self.blocks = nn.ModuleList(
            [
                CompositeBlock(
                    hidden_dim=config.hidden_dim,
                    physics_dim=physics_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.num_blocks)
            ]
        )

        # Heads
        self.aux_head = nn.Linear(config.hidden_dim, 1)
        self.final_head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x):
        # x: (Batch, Time, Features)

        # Extract physics features for injection (R, C, interactions)
        # These are sliced from the input tensor based on config indices
        physics_feats = x[:, :, self.physics_indices]

        # Initial Feature Extraction via Stem
        h = self.stem(x)

        aux_out = None

        # Process Blocks
        for i, block in enumerate(self.blocks):
            h = block(h, physics_feats)

            # Deep Supervision: Attach auxiliary head at specified block index
            if i == self.config.aux_head_block_idx:
                aux_out = self.aux_head(h)

        # Final Prediction
        final_out = self.final_head(h)

        # Squeeze last dimension to match target shape (Batch, Time)
        final_out = final_out.squeeze(-1)
        if aux_out is not None:
            aux_out = aux_out.squeeze(-1)

        return final_out, aux_out
